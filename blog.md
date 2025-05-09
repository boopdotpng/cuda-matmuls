# How to actually write fast GPU code?
This post is an in depth overview on GPU architecture and how to write performant GPU code. It covers execution hierarchy, memory layout, scheduling, memory access patterns, and basic profiling. The goal is to build enough knowledge to write a SGEMM (single precision general matrix multiply) kernel that achieves 50% of theoretical GPU FLOPS. 

The specifics in this guide, including naming and the specific capabilities of each SM are tailored to Nvidia's Blackwell generation of cards. However, this guide is applicable to all GPUs. 
## GPU Architecture Overview
This is a high level chart that shows the hierarchy of components in an Nvidia GPU. The GPC is the highest abstraction layer, and there are usually 6-12 per chip. 
<svg width="700" height="580" xmlns="http://www.w3.org/2000/svg" style="font-family: monospace; font-size: 14px">
  <!-- GPU Box -->
  <rect x="20" y="20" width="660" height="540" fill="#121212" stroke="#aaa" stroke-width="1.5"/>
  <text x="30" y="40" fill="#fff">GPU</text>

  <!-- GPC Box -->
  <rect x="40" y="60" width="620" height="320" fill="#2d2d4d" stroke="#aaa"/>
  <text x="50" y="80" fill="#fff">GPC (Graphics Processing Cluster)</text>

  <!-- TPC 1 -->
  <rect x="60" y="100" width="580" height="120" fill="#2d4d2d" stroke="#aaa"/>
  <text x="70" y="120" fill="#fff">TPC (Texture Processing Cluster)</text>
  <!-- SM Boxes in TPC 1 -->
  <rect x="80" y="130" width="260" height="80" fill="#4d2d2d" stroke="#aaa"/>
  <text x="90" y="150" fill="#fff">SM (Streaming Multiprocessor)</text>
  <text x="90" y="170" fill="#ddd">- 128 cores, 4 tensor cores</text>
  <rect x="360" y="130" width="260" height="80" fill="#4d2d2d" stroke="#aaa"/>
  <text x="370" y="150" fill="#fff">SM (Streaming Multiprocessor)</text>
  <text x="370" y="170" fill="#ddd">- 128 cores, 4 tensor cores</text>

  <!-- TPC 2 -->
  <rect x="60" y="240" width="580" height="120" fill="#2d4d2d" stroke="#aaa"/>
  <text x="70" y="260" fill="#fff">TPC (Texture Processing Cluster)</text>
  <!-- SM Boxes in TPC 2 -->
  <rect x="80" y="270" width="260" height="80" fill="#4d2d2d" stroke="#aaa"/>
  <text x="90" y="290" fill="#fff">SM (Streaming Multiprocessor)</text>
  <text x="90" y="310" fill="#ddd">- 128 cores, 4 tensor cores</text>
  <rect x="360" y="270" width="260" height="80" fill="#4d2d2d" stroke="#aaa"/>
  <text x="370" y="290" fill="#fff">SM (Streaming Multiprocessor)</text>
  <text x="370" y="310" fill="#ddd">- 128 cores, 4 tensor cores</text>

  <!-- L2 Cache Box -->
  <rect x="40" y="400" width="620" height="40" fill="#4d4d2d" stroke="#aaa"/>
  <text x="50" y="425" fill="#fff">L2 Cache — 70 MB on 4090, shared across all SMs</text>

  <!-- Global Memory Box -->
  <rect x="40" y="450" width="620" height="60" fill="#2d4d4d" stroke="#aaa"/>
  <text x="50" y="480" fill="#fff">Global Memory — 8–192 GB (GDDR6/GDDR7 or HBM), off-chip DRAM</text>
</svg>
If you want to see a more comprehensive review of GPU architecture check out [High Yield's](https://www.youtube.com/@HighYield) videos on YouTube. He does a great job of showing where each element is on the physical GPU die. 

The purpose of the GPCs and TPCs is to organize SMs (the main compute of the GPU) into modular blocks that have their own memory, cache, instruction dispatch, and texture units. Without this abstraction, there would be excessive contention for global resources and scaling the chip across product tiers would be much more difficult. 

GPCs in traditional consumer GPUs also handle rasterization and graphics functions. In compute-only GPUs like the Nvidia H100, they may be optimized for throughput. For machine learning oriented workloads, this almost never comes into the picture. We're focused entirely on the SMs.
### Streaming Multiprocessors 
There are a lot of individual components that make up an SM: 

| Element                | Notes                                                                                                              | Count / Size Per SM |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------- |
| **Compute**            |                                                                                                                    |                     |
| CUDA cores             | Scalar ALUs that can execute one FP32 or INT32 instruction per clock cycle, per core, on a single operand. <br>    | 128                 |
| Tensor cores           | Accelerates small matrix multiply-accumulate ops using mixed precision (FP16, BF16, TF32).                         | 4                   |
| Special Function Units | Handles transcendental and high-latency functions: sin, cos, exp, sqrt, etc.                                       | 4                   |
| Warp schedulers        | Manages instruction dispatch for one warp (32 threads) per cycle, directing execution to available CUDA cores.<br> | 4                   |
| Load/Store units       | Interface for memory ops (load, store). Routes data to/from memory hierarchy.                                      | 4–8                 |
|                        |                                                                                                                    |                     |
| **Memory**             |                                                                                                                    |                     |
| Register file          | Fast, per-thread memory used for all intermediate values. Like CPU registers, but all 32-bit.                      | 128–256 KB          |
| Shared memory/L1 cache | Low-latency, per-SM memory. Shared memory is stored in L1 cache and is managed by the programmer.                  | 64–164 KB           |
Most if not all of the compute on a GPU is done by CUDA cores. Some mixed precision datatypes (fp16, bf16, tf32, etc) are offloaded to other units within the SM (tensor cores for example), along with all exp, sin, cos-adjacent computations (on SFUs). 
### Execution model 
The GPU execution model follows a hierarchy; from bottom to top: 

A **thread** is the smallest unit of execution on the GPU. Every thread runs its own instance of the kernel function, with its operations independently scheduled on CUDA cores.

A **warp** is a fixed group of 32 threads (all from the same block) that are executed in lockstep under the SIMT (single instruction multiple thread) model. Each SM has 4 warp schedulers, each capable of issuing one instruction per cycle to a warp. In practice, the SM can track and switch between dozens of warps (active or stalled), depending on occupancy. This is crucial for mitigating memory latency. If memory access is slow for one warp, it can be put aside and executed later once the data is ready. Context switching like this is extremely cheap on a GPU. It's also important to note that memory access requests are done per warp level, not per thread.

A **block** is a group of threads (up to 1024) that execute together and share memory. Blocks are assigned to individual SMs, and multiple blocks can be scheduled on the same SM if there are enough available resources (dependent on register and shared memory usage). The number of active threads and blocks per SM is known as occupancy. 

A **grid** is a collection of blocks that covers all blocks and threads launched by the kernel and spans the entire GPU. Blocks within a grid cannot communicate or share memory with each other. 

This is how each part of the execution model maps to CUDA terms. Each parameter in table is of type `dims3`, meaning it has 3 dimensions (x, y, z). 

| Parameter | Notes                               |
| --------- | ----------------------------------- |
| blockIdx  | Which block is this thread in?      |
| blockDim  | How many threads are in each block? |
| threadIdx | Where in the block is this thread?  |
| gridDim   | How many total blocks are there?    |
<svg width="700" height="520" xmlns="http://www.w3.org/2000/svg" style="font-family: monospace; font-size: 14px">
  <!-- Outer Grid Box -->
  <rect x="20" y="20" width="660" height="470" fill="#121212" stroke="#aaa" stroke-width="1.5"/>
  <text x="30" y="40" fill="#fff">grid (1d) gridDim = 3</text>
  
  <!-- Block 0 -->
  <rect x="60" y="60" width="600" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="70" y="80" fill="#fff">Block 0 (x=0) blockDim = 3</text>
  <!-- Threads in Block 0 -->
  <rect x="100" y="100" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="110" y="130" fill="#fff">T0</text>
  <rect x="280" y="100" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="290" y="130" fill="#fff">T1</text>
  <rect x="460" y="100" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="470" y="130" fill="#fff">T2(threadIdx = 2)</text>

  <!-- Block 1 -->
  <rect x="60" y="200" width="600" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="70" y="220" fill="#fff">Block 1 (x=1) blockIdx = 1 for all threads</text>
  <!-- Threads in Block 1 -->
  <rect x="100" y="240" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="110" y="270" fill="#fff">T0 (gid = 3)</text>
  <rect x="280" y="240" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="290" y="270" fill="#fff">T1(threadIdx = 1)</text>
  <rect x="460" y="240" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="470" y="270" fill="#fff">T2</text>

  <!-- Block 2 -->
  <rect x="60" y="340" width="600" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="70" y="360" fill="#fff">Block 2</text>
  <!-- Threads in Block 2 -->
  <rect x="100" y="380" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="110" y="410" fill="#fff">T0 (gid = 6)</text>
  <rect x="280" y="380" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="290" y="410" fill="#fff">T1</text>
  <rect x="460" y="380" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="470" y="410" fill="#fff">T2</text>
</svg>
### Memory hierarchy

| Memory type          | Latency (cycles) | Bandwidth  |
| -------------------- | ---------------- | ---------- |
| Global (GDDR or HBM) | 400-800          | 0.8-1 TB/s |
| L2 cache             | 100-200          | 1-2 TB/s   |
| L1 cache/shared      | 20-40            | 1-2 TB/s   |
| Register file        | 1-4              | >10 TB/s   |
Accessing memory, especially global memory, is often orders of magnitude more expensive than compute. It's often the major bottleneck for performance. In the matrix multiplication examples later, you'll see that the actual compute doesn't change at all. We only make the memory access faster each kernel.  
#### Global memory 
Global memory is accessible to all SMs and represents the largest but slowest memory region on the GPU, typically implemented as off-chip GDDR6 or GDDR7 DRAM. It serves as the main interface between CPU and GPU, storing model weights, input datasets, and output buffers.

When you call `cudaMalloc`, the pointer returned points to a region in this memory.
#### L2 cache
L2 cache is a unified, on chip cache shared by all SMs. It sits between global memory and the SMs, buffering data to reduce access latency and minimize redundant memory traffic. 
#### L1 cache / shared memory 
L1 cache is fast, low latency cache local to each SM and shares physical space with shared memory. Shared memory can be explicitly allocated in a kernel using the `__shared__` keyword and is accessible only within the same block. 
#### Register file
Each SM has a large bank of 32-bit registers (around 128 KB) divided among its active threads. Registers are the fastest form of memory and are private to each thread.

The number of registers used per thread directly constrains occupancy: more registers per thread mean fewer threads per SM. At the low level (PTX or SASS), registers fall into categories (general-purpose, predicate, special), but these details are rarely relevant outside hand-tuned kernel work.
#### Memory coalescing 
Memory access on GPUs occurs at the warp level—each warp of 32 threads issues memory requests together. When threads access global memory, the hardware attempts to combine their individual requests into as few large transactions as possible, typically aligned 128-byte.

Coalescing is most efficient when threads access consecutive and properly aligned addresses (a float array accessed linearly). In such cases, the entire warp can be served with a single 128-byte transaction. When access patterns are irregular, misaligned, or sparse, the warp may generate multiple transactions, each with higher latency and lower throughput.

Efficient memory coalescing is key to reducing bandwidth waste and hiding global memory latency. We’ll revisit this in detail during the matrix multiplication section.
## Basic kernel example
A kernel launch consists of the following: 
1. A GPU kernel function
2. Number of blocks 
3. Number of threads per block
4. The data you want to write 
### 1D Dispatch 
 Consider the following kernel that adds two arrays `A+B` and stores the output in another array `C`. We'll assume that `len(a) = len(b) = len(c) = 1000`.
```cpp
__global__ void add(const float *a, const float *b, float *c) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= 1000) return;
	c[gid] = a[gid] + b[gid];
}
```

In this kernel, the first thread calculates `c[0] = a[0] + b[0]`, the second `c[1] = a[1] + b[1]`, and so on. This requires us to launch 1000 threads. 

To launch this kernel, we need to determine the launch configuration -- specifically the number of blocks and threads per block. 

The typical approach is to choose the number of threads per block, and then compute how many blocks are needed to cover the entire kernel. In this example, we'll choose 128 threads per block, which means we'll need 8 blocks to cover all 1000 threads (128 * 8 = 1024).  

So, for this launch: 
- gridDim = (8, 1, 1) (8 total blocks)
- blockDim = (128, 1, 1) (128 threads per block) 
- threadIdx = (0..127, 1, 1)
- blockIdx = (0..7, 1, 1)

The extra `1` dimensions are added by default if you don't specify them. Remember that these values are 3d (x, y, z). Going forward, if the dimension doesn't exist, I won't mention it. 

Looking at the `gid` calculation:
```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```

We get the global id by multiplying which block the thread is in by how many total blocks there are, and then adding the position of the current thread in the block. This gives us the global position of the thread, relative to every other thread in the dispatch. Refer back to [[#Execution model]] for a visual.

All the parameters listed here are actually three dimensional, but since our data in this kernel is 1d, we only use one dimension. 2d and 3d dispatches are just abstractions over a 1d dispatch, and mostly exist to make indexing more convenient when you're operating on matrices. 

#### Over-launching and powers of 2
Due to the way GPU hardware is designed, you should use a power-of-two number for threads per block. This avoids having a partially unfilled warp, which hurts throughput. Even though this isn't strictly necessary, powers of two have several other advantages: 
- Promotes coalesced memory accesss, since addresses are more likely to be aligned and regularly spaced
- Enables faster index math, as bit shifting is cheaper than division or modulo
- Simplifies tiling (especially for a tiled matrix multiplication, which we will see later)

To prevent these extra threads from accessing out-of-bounds memory, we add a guard that exits the kernel if the thread number is more than 999.
```cpp
if (gid >= 1000) return;
```
#### Why 1d dispatch breaks down for 2d data
This style of indexing works very well when your data is 1 dimensional, but falls apart fast when you're working with 2d structures like matrices or images. Consider a `32x32` matrix stored in [row-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order) order. We calculate the `gid` value the same way as last time, but now, since our data is 2d, we have to manually unflatten the index into a (col, row) pair. 
```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
int row = gid / width;
int col = gid % width;
```
This calculation wastes cycles on the GPU and introduces extra complexity to every kernel. It also makes the structure of the data hard to reason about. 2d dispatching aims to make this much simpler.
### Visualizing a 2d dispatch
To visualize 2d thread dispatch, we will write a kernel that records each thread's global (x,y) coordinate into a 2d matrix.
```cpp
__global__ void record_thread_coords(int* coords, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x; 
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  int idx = row * width + col; // flattened row-major index
  
  coords[2 * idx + 0] = col; // x coordinate
  coords[2 * idx + 1] = row; // y coordinate
}
```

The shape of `coords` is `(2,6,4)` represented as a flat `int[48]`. For each cell in the `6,4` grid, we store 2 coordinates (`x,y`). We need to launch 24 threads to cover the entire grid. Consider the following arrangement, where `blockDim = (2,2)`, `gridDim = (2,3)`, and `total_threads = 2*2*2*3 = 24`:  
<svg width="560" height="580" xmlns="http://www.w3.org/2000/svg" style="font-family: monospace; font-size: 14px">
  <!-- Outer Grid Box -->
  <rect x="20" y="20" width="520" height="540" fill="#121212" stroke="#aaa" stroke-width="1.5"/>
  <text x="30" y="40" fill="#fff">grid (2d) gridDim = (2,3)</text>

  <!-- Block (0,0) -->
  <rect x="40" y="60" width="180" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="50" y="80" fill="#fff">Block (0,0)</text>
  <rect x="55" y="95" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="60" y="115" fill="#fff">(0,0)</text>
  <rect x="135" y="95" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="140" y="115" fill="#fff">(0,1)</text>
  <rect x="55" y="135" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="60" y="155" fill="#fff">(1,0)</text>
  <rect x="135" y="135" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="140" y="155" fill="#fff">(1,1)</text>

  <!-- Block (1,0) -->
  <rect x="240" y="60" width="180" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="250" y="80" fill="#fff">Block (1,0)</text>
  <rect x="255" y="95" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="260" y="115" fill="#fff">(0,2)</text>
  <rect x="335" y="95" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="340" y="115" fill="#fff">(0,3)</text>
  <rect x="255" y="135" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="260" y="155" fill="#fff">(1,2)</text>
  <rect x="335" y="135" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="340" y="155" fill="#fff">(1,3)</text>

  <!-- Block (0,1) -->
  <rect x="40" y="200" width="180" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="50" y="220" fill="#fff">Block (0,1)</text>
  <rect x="55" y="235" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="60" y="255" fill="#fff">(2,0)</text>
  <rect x="135" y="235" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="140" y="255" fill="#fff">(2,1)</text>
  <rect x="55" y="275" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="60" y="295" fill="#fff">(3,0)</text>
  <rect x="135" y="275" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="140" y="295" fill="#fff">(3,1)</text>

  <!-- Block (1,1) -->
  <rect x="240" y="200" width="180" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="250" y="220" fill="#fff">Block (1,1)</text>
  <rect x="255" y="235" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="260" y="255" fill="#fff">(2,2)</text>
  <rect x="335" y="235" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="340" y="255" fill="#fff">(2,3)</text>
  <rect x="255" y="275" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="260" y="295" fill="#fff">(3,2)</text>
  <rect x="335" y="275" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="340" y="295" fill="#fff">(3,3)</text>

  <!-- Block (0,2) -->
  <rect x="40" y="340" width="180" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="50" y="360" fill="#fff">Block (0,2)</text>
  <rect x="55" y="375" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="60" y="395" fill="#fff">(4,0)</text>
  <rect x="135" y="375" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="140" y="395" fill="#fff">(4,1)</text>
  <rect x="55" y="415" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="60" y="435" fill="#fff">(5,0)</text>
  <rect x="135" y="415" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="140" y="435" fill="#fff">(5,1)</text>

  <!-- Block (1,2) -->
  <rect x="240" y="340" width="180" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="250" y="360" fill="#fff">Block (1,2)</text>
  <rect x="255" y="375" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="260" y="395" fill="#fff">(4,2)</text>
  <rect x="335" y="375" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="340" y="395" fill="#fff">(4,3)</text>
  <rect x="255" y="415" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="260" y="435" fill="#fff">(5,2)</text>
  <rect x="335" y="415" width="70" height="35" fill="#2d4d2d" stroke="#aaa"/>
  <text x="340" y="435" fill="#fff">(5,3)</text>
</svg>
To illustrate the `col` and `row` calculations, let's go through the kernel for thread `2,3`.
`blockIdx = (1,1)` `blockDim = (2,2)` `threadIdx = (0,1)`. 
```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x; 
int row = blockIdx.y * blockDim.y + threadIdx.y;
```
The x component of the global id is `1 * 2 + 0 = 2`.
The y component of the global id is `1 * 2 + 1 = 3`.
 ```
Output (coords array): 
(0,0) (1,0) (2,0) (3,0)
(0,1) (1,1) (2,1) (3,1)
(0,2) (1,2) (2,2) (3,2)
(0,3) (1,3) (2,3) (3,3)
(0,4) (1,4) (2,4) (3,4)
(0,5) (1,5) (2,5) (3,5)
```

The printed coordinates are in `(col, row)` format, where `col` corresponds to the horizontal `x` axis and `row` to the vertical `y` axis. This mirrors standard 2d matrix conventions, where the first dimension indexes rows (`y`) and the second indexes columns (`x`). each entry `(x, y)` represents the global coordinates of a thread in the grid.

You can see this example at [cuda-matmuls/misc/2d_dispatch_viz.cu](https://github.com/boopdotpng/cuda-matmuls/blob/master/misc/2d_dispatch_viz.cu).
## Matrix multiplication

### Theoretical matrix multiplication performance
Matrix multiplication is one of the most common dense compute kernels in machine learning. This section covers a series of increasingly optimized matrix multiplication kernels. [This](http://matrixmultiplication.xyz/) is the best way to visualize it. 

To find out how fast a matrix multiplication kernel can be on your GPU, you can use the `cuBLAS` library, which contains highly optimized kernels written by Nvidia. These kernels are fine tuned to extract the maximum performance from the hardware; it's extremely difficult to outperform a `cuBLAS` kernel.

All of the examples going forward will be multiplying two 4096x4096 matrices in single precision (SGEMM). 

Performance is measured in TFLOPS (trillion floating point operations per second). In order to calculate the theoretical maximum FP32 performance for my GPU (a 5070 Ti): 
- 70 SMs * 128 Cores per SM = 8960 Cuda Cores
- Each Cuda core performs 2 operations per clock cycle (FMA = fused multiply-add) 
- Boost clock: 2.45 GHz  = 2.45 * 10^9 cycles per second
- Equals approximately 44 TFLOPS. 

Now, let's estimate the number of operations required to multiply two square matrices of size 4096. Each of the `N^2` cells in matrix C requires a dot product between a row of A and a column of B, consisting of `N` multiplications and `N` additions. That’s `2N` floating point operations per entry, yielding a total of `2*N^3` FLOPs.

So the total number of operations is `2*4096^3 = 137,438,953,472`. 

TFLOPS = `(2*4096^3) / (execution time in seconds × 10^12)`

The `cuBLAS` kernel hovers around 34 TFLOPS on my GPU (77% of theoretical). You'll never get the advertised theoretical performance due to warp scheduling, memory access patterns, and many other factors. We'll compare all future kernels to the 34 TFLOPS max instead of the theoretical 44 TFLOPS because it's a much more realistic estimate of how fast our kernel could be.
### Matrix multiplication on CPU
The most straightforward `N*N` square matrix multiplication goes like this: 
```cpp
float A[N][N], B[N][N], C[N][N];
for (int i = 0; i < N; i++)
  for (int j = 0; j < N; j++) {
    float acc = 0.0;
    for (int k = 0; k < N; k++)
	  acc += A[i][k] * B[k][j];
    C[i][j] = acc;
}
```

For each output cell in C, we calculate the [dot product](https://en.wikipedia.org/wiki/Dot_product) of row `i` from matrix A and column `j` from matrix B. This is an incredibly slow way to multiply matrices: the time complexity is `O(n^3)` and it only achieves around 0.019 TFLOPS for a 1024x1024 matrix. This example is missing SIMD instructions, use of multiple cores, cache-friendly access for B (memory access not coalesced), to name a few. 

Numpy delegates matrix multiplication to high performance BLAS libraries, which use multithreading and SIMD. They're extremely optimized, and a good way to figure out how far you are from the theoretical performance of your CPU. 
```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python -c "import numpy as np, time; N=4096; A=np.random.rand(N,N).astype(np.float32); B=np.random.rand(N,N).astype(np.float32); t0=time.time(); C=A@B; t1=time.time(); flops=2*N**3; dt=t1-t0; print(f'Time: {dt:.4f} s, GFLOPS: {flops/dt/1e9:.2f}')"
```
This gets around 0.3 TFLOPS on my Ryzen 7 9700x (one core). For multi-threaded performance, just remove the environment variables (1.4 TFLOPS).
### The simplest GPU matrix multiplication 
This code is a copy of the CPU matrix multiplication code, with the outer loops replaced by thread indexing. Each thread computes a single element in the output matrix `C`. 

Since the matrix is 4096×4096, we launch 16,777,216 threads total. Using 256 threads per block (`blockDim = (16, 16)`), we require 65,536 blocks (`gridDim = (256, 256)`).

The launch configuration is two dimensional: 
`blockDim = (16,16)`  -> 256 threads per block,
`gridDim = (256,256)`  -> 65,536 total blocks.
```cpp
__global__ void matmul(const float *a, const float *b, float *c) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (uint i = 0; i < 4096; ++i)
        sum += a[row * 4096 + i] * b[col * 4096 + i];
    c[row * 4096 + col] = sum;
}
```

This hovers around 0.68 TFLOPS (50x slower than theoretical). Interestingly, this implementation is slower than Numpy's multithreaded CPU matrix multiplication, which means we're not utilizing the GPU effectively. 
#### Profiling kernels
`ncu` is the one of the best ways to understand what the bottleneck is for a particular kernel. You can see SM throughput, number of cycles, DRAM bandwidth, and a lot of other important statistics. Profiling this kernel using `ncu` : 

| Metric                     | Value       | Unit   |
| -------------------------- | ----------- | ------ |
| Compute (SM) Throughput    | 21.79       | %      |
| Memory Throughput          | 98.06       | %      |
| DRAM Throughput            | 7.52        | %      |
| Elapsed Cycles             | 5.63281e+08 | cycles |
| Average SM Active Cycles   | 5.62532e+08 | cycles |
| Average L1 Active Cycles   | 5.62532e+08 | cycles |
| Average L2 Active Cycles   | 5.04343e+08 | cycles |
| Average DRAM Active Cycles | 2.54577e+08 | cycles |
The key takeaways here are that: 
- Compute throughput is only 22%, meaning nearly 80% of cycles are spent not executing useful instructions 
- DRAM throughput is only 7.52%, which implies most of the traffic is stalled or wasted 
- L2 sees significantly fewer cycles than L1. Since L2 sits between global memory and the SMs, this means that the global memory accesses are not being cached. 
- DRAM was active for ~45% of the kernel duration, meaning nearly **half** the time is spent waiting on global memory. 
This kernel is bottlenecked by inefficient global memory accesses. 
#### Fixing the memory access issue
The root cause of the memory issues in the previous kernel is this line: 
```cpp
sum += a[row * 4096 + i] * b[col * 4096 + i];
```

The way we access A is row-major. This is inherently good for memory coalescing, since you're reading strictly left to right. All the values are next to each other in memory.

B, on the other hand access columns in a row-major array, which is very bad for coalescing.  
```
thread 0 → b[0 * 4096 + i] → b[i]
thread 1 → b[1 * 4096 + i] → b[4096 + i]
thread 2 → b[2 * 4096 + i] → b[8192 + i]
...
```
Each thread accesses a float 4096 elements away from the adjacent thread. Within a warp, this creates a lot of separate memory transactions, which we saw evidence of in the `ncu` report. 

The most straightforward solution here is to transpose B. If you transpose B, the rows and columns are switched. Then, we can access the columns of B the same way we access the rows of A. Now, both A and B are accessed row-wise, reducing the overall memory transactions greatly.  

```cpp
 sum += a[row * 4096 + i] * b[i * 4096 + col];
```

Below is the table for the coalesced memory accesses for the new kernel starting at the first thread.

| threadIdx.x | col | Access `A[0][0]` (same for all) | Access `B[0][col]` | Address in B |
| ----------- | --- | ------------------------------- | ------------------ | ------------ |
| 0           | 0   | `a[0 * 4096 + 0]` = `a[0]`      | `b[0 * 4096 + 0]`  | `b[0]`       |
| 1           | 1   | `a[0]`                          | `b[0 * 4096 + 1]`  | `b[1]`       |
| 2           | 2   | `a[0]`                          | `b[0 * 4096 + 2]`  | `b[2]`       |
| ...         | ... | `a[0]`                          | ...                | ...          |
| 15          | 15  | `a[0]`                          | `b[0 * 4096 + 15]` | `b[15]`      |

With B transposed, the kernel hovers around 2.7 TFLOPS (~13x slower than theoretical and a ~4x speedup from the previous kernel).  
#### NCU comparison table
Here is a table comparing profiling results from the two kernels. The first column is the new kernel where we transposed B, and the second column is the kernel that we profiled earlier.

| Metric              | matmul_B_transposed | matmul_B_original | Unit  |
| ------------------- | ------------------- | ----------------- | ----- |
| SM Throughput       | 92.33               | 21.79             | %     |
| DRAM Throughput     | 33.58               | 7.52              | %     |
| L2 Cache Throughput | 28.38               | 9.01              | %     |
| Duration            | 57.95               | 245.92            | ms    |
| Total DRAM Cycles   | 6.39e+09            | 2.71e+10          | cycle |
| Total L2 Cycles     | 2.86e+09            | 1.21e+10          | cycle |
The key differences to note: 
- SM throughput increased to 92% (~4.2x higher), indicating that the GPU is now spending most of its time performing computations rather than stalling on memory accesses.
- Memory system cycles have gone down almost 5x.
- L2 cache throughput has gone up. Since we made our memory access more linear and predictable, more of our global reads can be cached and reused.
- DRAM throughput is up ~4.5x. By transposing matrix B, we enabled coalesced memory access, allowing adjacent threads to read adjacent memory locations.
### Tiled matmul 
In the previous kernel, every thread computed one element from the output matrix by loading: 
- A full row of `A` from global memory
- A full column of `B` from global memory
```cpp
for (uint i = 0; i < N; ++i)
	sum += a[row * N + i] * b[i * N + col];
```

Since many threads access overlapping rows and columns, the same values are repeatedly fetched from global memory -- thousands of times. This is wasteful, because values fetched here are not reused across threads.

**Solution: shared memory tiling** 
Shared memory (fast, on chip L1) allows a thread block to load and reuse data. Instead of every thread individually accessing global memory: 
1. We divide the matrix into tiles (16x16 blocks) 
2. Every block loads one tile of A and one tile of B into shared memory
3. All threads in that block compute partial products using only shared memory
4. This process is repeated until the full dot product is accumulated 

By doing this, each value from `A` and `B` is loaded from global memory once per tile instead of once per thread. This drastically reduces the amount of global memory accesses that our kernel performs. 

This [twitter](https://x.com/Hesamation/status/1920141361531040152) post does a great job visualizing data accesses and cache hits for this method. 

The launch parameters for this kernel are the same as the previous one. We're still launching one thread per output element of C, and since each block is responsible for a `16x16` tile, we have to launch 256 threads per block. `gridDim = (N/16, N/16)`, same as before. 
```cpp
__global__ void tiled_matmul(const float *a, const float *b, float *c) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float Asub[TILE][TILE];
    __shared__ float Bsub[TILE][TILE];

    float acc = 0.0f;
    for (int k0 = 0; k0 < N; k0 += TILE) {
      Asub[threadIdx.y][threadIdx.x] = a[row * N + (k0 + threadIdx.x)];
      Bsub[threadIdx.y][threadIdx.x] = b[(k0 + threadIdx.y) * N + col];
      __syncthreads();

      for (int k = 0; k < TILE; ++k)
        acc += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
      
      __syncthreads();
    }

    c[row * N + col] = acc;
}
```


This kernel hovers around 3.7 TFLOPS (~9x slower than theoretical) with 16x16 tiles.
### More optimizations



## References 

This was partially inspired by some George Hotz streams I watched:
- [how do GPUs work?](https://youtu.be/OUzm06YaUsI) 
- [can you multiply a matrix?](https://youtu.be/VgSQ1GOC86s)). 

All the code in this post can be found on [GitHub](https://github.com/boopdotpng/cuda-matmuls).

For another perspective on CUDA and GPU architecture, see the guide at [modal.com](https://modal.com/gpu-glossary). 

[Nvidia blackwell architecture whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture).

[High Yield: 5090 deep dive](https://youtu.be/rCwgAGG2sZQ)

[Twitter](https://x.com/Hesamation/status/1920141361531040152) visualization of how data and cache is accessed for a tiled matrix multiply kernel. 