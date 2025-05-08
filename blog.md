# How to actually write fast GPU code?
This post is an in depth overview on GPU architecture and how to write performant GPU code. It covers execution hierarchy, memory layout, scheduling, memory access patterns, and basic profiling. The goal is to build enough knowledge to write a SGEMM (single precision general matrix multiply) kernel that achieves more than 10 TFLOPS on any high-end GPU. 

The specifics in this guide, including naming and the specific capabilities of each SM are tailored to Nvidia's Blackwell generation of cards. However, most of this guide is still applicable to GPUs in general. The fundamental concepts and architecture remain mostly the same. 
## GPU Architecture Overview
This is a high level chart that shows the hierarchy of components in a GPU. The GPC is the highest abstraction layer at the top of the GPU, and there are usually 6-12 per chip. 
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
[High Yield](https://www.youtube.com/@HighYield) makes great videos on chip architecture (the 5090 video is particularly relevant here). In these videos, you can roughly see the blocks on the actual GPU silicon that correspond to the caches, memory controllers, PCIe controllers, etc.

The purpose of the GPC and TPC is to organize SMs (the main compute of the GPU) into modular blocks that have their own memory, cache, instruction dispatch, and texture units. Without this abstraction, there would be excessive contention for global resources and scaling the chip across product tiers would be much more difficult. 

GPCs in traditional consumer GPUs also handle rasterization and graphics functions. In compute-only GPUs like the Nvidia H100, they may be optimized for throughput. 
### Streaming Multiprocessors 
The individual components in an SM can be split roughly into Compute and Memory. 

| Element                | Notes                                                                                                              | Count / Size Per SM |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------- |
| **Compute Units**      |                                                                                                                    |                     |
| CUDA cores             | Scalar ALUs that can execute one FP32 or INT32 instruction per clock cycle, per core, on a single operand. <br>    | 128                 |
| Tensor cores           | Accelerates small matrix multiply-accumulate ops using mixed precision (FP16, BF16, TF32).                         | 4                   |
| Special Function Units | Handles transcendental and high-latency functions: sin, cos, exp, sqrt, etc.                                       | 4                   |
| Warp schedulers        | Manages instruction dispatch for one warp (32 threads) per cycle, directing execution to available CUDA cores.<br> | 4                   |
| Load/Store units       | Interface for memory ops (load, store). Routes data to/from memory hierarchy.                                      | 4–8                 |
|                        |                                                                                                                    |                     |
| **Memory Units**       |                                                                                                                    |                     |
| Register file          | Fast, per-thread memory used for all intermediate values. Like CPU registers, but all 32-bit.                      | 128–256 KB          |
| Shared memory/L1 cache | Low-latency, per-SM memory. Shared memory is stored in L1 cache and is managed by the programmer.                  | 64–164 KB           |
Most, if not all of the compute on a GPU is done by CUDA cores. Some very specific datatypes (fp16, bf16, tf32, etc) are offloaded to other parts of the SM (tensor cores for example), along with all exp, sin, cos-adjacent computations (SFUs). 

All instructions are ultimately scheduled at the warp level. The four warp schedulers per SM each handle one 32 thread warp independently. 

Each SM contains its own registers, shared memory, and access to the L1 cache. These memory units are part of a greater hierarchy that defines how memory is accessed inside of a kernel.
### Memory Hierarchy

| Memory type          | Latency (cycles) | Bandwidth  | notes                        |
| -------------------- | ---------------- | ---------- | ---------------------------- |
| Global (GDDR or HBM) | 400-800          | 0.8-1 TB/s | high latency, off chip dram  |
| L2 cache             | 100-200          | 1-2 TB/s   | shared between SMs           |
| L1 cache/shared      | 20-40            | 1-2 TB/s   | per-sm, fast for threads     |
| Register file        | 1-4              | >10 TB/s   | per-core, extremely fast<br> |
Accessing memory is far more time intensive than the actual compute. The largest contributor to this latency is global memory. These bandwidth numbers represent peak throughput across all SMs and varies by architecture. 
#### Global memory 
Global memory is accessible to all SMs and represents the largest but slowest memory region on the GPU, typically implemented as off-chip GDDR6 or GDDR7 DRAM. It serves as the main interface between CPU and GPU, storing model weights, input datasets, and output buffers.

When you call `cudaMalloc`, the pointer returned points to a region in this memory.
#### L2 cache
L2 cache is a unified, on chip cache shared by all SMs. It sits between global memory and the SMs, buffering data to reduce access latency and minimize redundant memory traffic. 
#### L1 cache / shared memory 
L1 cache is fast, low latency cache local to each SM and typically shares physical space with shared memory. Shared memory is explicitly allocated in a kernel using the `__shared__` keyword inside a kernel and is accessible only to threads within the same block. Because of its speed and block-level scope, it's often used to stage data from global memory, reducing costly global reads. This is a critical tool for optimizing memory-bound kernels.
#### Register file
These closely resemble registers on a CPU, except there are many times more per core in a GPU. Total capacity per SM is around 128 KB (architecture-dependent), which is divided across all active threads. Each thread gets its own private set of registers, and the number used per thread directly affects how many threads can run concurrently on an SM. There are different kinds of registers (general-purpose, predicate, special), but these distinctions only become relevant when examining PTX (NVIDIA’s intermediate representation) or SASS (its assembly-level counterpart).
#### Memory coalescing 
Think of every memory access as an independent transaction.  When a warp of 32 threads accesses memory, the GPU attempts to merge (coalesce) those individual accesses into a few larger transactions. This is most effective when threads access memory in a regular, aligned pattern. Scattered accesses result in many memory transactions, which adds latency and increases the time we have to wait for data to be available. We'll explore this more in the matrix multiplication section.
### Execution model 

The GPU execution model follows a hierarchy; from bottom to top: 

A **thread** is the smallest unit of execution on the GPU. Every thread runs its own instance of the kernel function, with its operations independently scheduled on CUDA cores.

A **warp** is a fixed group of 32 threads (all from the same block) that are executed in lockstep under the SIMT (single instruction multiple thread) model. Each SM has 4 warp schedulers, each capable of issuing one instruction per cycle to a warp. In practice, the SM can track and switch between dozens of warps (active or stalled), depending on occupancy. This is crucial for mitigating memory latency. If memory access is slower for one warp, it can be put aside and executed later once the data is ready. Context switching like this is extremely cheap on a GPU.

A **block** is a group of threads (up to 1024) that execute together and share memory. Blocks are assigned to individual SMs, and multiple blocks can be scheduled on the same SM if there are enough available resources (dependent on register and shared memory usage). The number of active threads and blocks per SM is known as occupancy. 

A **grid** is a collection of blocks that covers all blocks and threads launched by the kernel and spans the entire GPU. Blocks within a grid cannot communicate or share memory with each other. 
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

The typical approach is to choose the number of threads per block, and then compute how many blocks are needed to cover the entire kernel. In this example, we select 128 threads per block, which means we'll need 8 blocks to cover all 1000 threads (128 * 8 = 1024). 

**A note about overlaunching and powers of two:**
Due to the way GPU hardware is designed, there are a couple important rules to keep in mind when deciding the number of threads per block. Threads are executed in groups of warps, consisting of 32 threads. To take full advantage of this, it's best to use a power-of-two number of threads per block. This avoids partially filled warps, which hurts throughput. 

However, following this rule results in launching more threads than are necessary. In this example, we launched 24 extra threads (1024 - 1000). To prevent these threads from accessing out-of-bounds memory, we add a guard that exits the kernel if the thread number is more than 999.

Back to the `gid` calculation:
The GPU runtime injects couple parameters that tell each thread where it is in the grid, so you can determine where each thread should be writing. 

| Parameter | Notes                               | Example for add kernel |     |
| --------- | ----------------------------------- | ---------------------- | --- |
| blockIdx  | Which block is this thread in?      | 0 to 7                 |     |
| blockDim  | How many threads are in each block? | 128                    |     |
| threadIdx | Where in the block is this thread?  | 0 to 127               |     |
| gridDim   | How many total blocks are there?    | 8                      |     |
<svg width="700" height="520" xmlns="http://www.w3.org/2000/svg" style="font-family: monospace; font-size: 14px">
  <!-- Outer Grid Box -->
  <rect x="20" y="20" width="660" height="470" fill="#121212" stroke="#aaa" stroke-width="1.5"/>
  <text x="30" y="40" fill="#fff">grid (1d) gridDim = 3</text>
  
  <!-- Block 0 -->
  <rect x="60" y="60" width="600" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="70" y="80" fill="#fff">Block 0 (x=0) blockDim = 3</text>
  <!-- Threads in Block 0 -->
  <rect x="100" y="100" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="110" y="130" fill="#fff">T0 (gid = 0)</text>
  <rect x="280" y="100" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="290" y="130" fill="#fff">T1</text>
  <rect x="460" y="100" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="470" y="130" fill="#fff">T2 (threadIdx = 2)</text>

  <!-- Block 1 -->
  <rect x="60" y="200" width="600" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="70" y="220" fill="#fff">Block 1 (x=1) blockIdx = 1</text>
  <!-- Threads in Block 1 -->
  <rect x="100" y="240" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="110" y="270" fill="#fff">T0 (gid = 3)</text>
  <rect x="280" y="240" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="290" y="270" fill="#fff">T1 (threadIdx = 1)</text>
  <rect x="460" y="240" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="470" y="270" fill="#fff">T2</text>

  <!-- Block 2 -->
  <rect x="60" y="340" width="600" height="120" fill="#2d2d4d" stroke="#aaa"/>
  <text x="70" y="360" fill="#fff">Block 2</text>
  <!-- Threads in Block 2 -->
  <rect x="100" y="380" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="110" y="410" fill="#fff">T0 (gid = 6)</text>
  <rect x="280" y="380" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="290" y="410" fill="#fff">T1 (gid = 7)</text>
  <rect x="460" y="380" width="160" height="60" fill="#2d4d2d" stroke="#aaa"/>
  <text x="470" y="410" fill="#fff">T2 (gid = 8)</text>
</svg>
```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```
We get the global id by multiplying which block the thread is in by how many total blocks there are, and then adding the position of the current thread in the block. This gives us the global position of the thread, relative to every other thread in the dispatch. 

All the parameters listed here are actually three dimensional, but for this example only one dimension (x) is necessary. 2d and 3d dispatches are just abstractions over a 1d dispatch, and mostly exists to make indexing more convenient when you're operating on matrices, as we will see later. 
### Visualizing a 2d dispatch
To visualize 2d thread dispatch, we write a kernel that records each thread's global (x,y) coordinates into a 2d matrix.
```cpp
__global__ void record_thread_coords(int* coords, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x; 
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  int idx = row * width + col; // flattened row-major index
  
  coords[2 * idx + 0] = col; // x coordinate
  coords[2 * idx + 1] = row; // y coordinate
}
```

The shape of `coords` is `(2,6,4)` represented as a flat `int[48]`. For each square in the `6,4` grid, we store 2 coordinates (`x,y`). We need to launch 24 threads to cover the entire grid. Consider the following arrangement, where `blockDim = (2,2)`, `gridDim = (2,3)`, and `total_threads = 2*2*2*3 = 24`:  
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
To illustrate the `gidx` and `gidy` calculations, let's go through the kernel for thread `2,3`.
`blockIdx = (1,1)` `blockDim = (2,2)` `threadIdx = (0,1)`. 
The x component of the global id is `1 * 2 + 0 = 2`.
The y component of the global id is `1 * 2 + 1 = 3`.

Now, you can calculate the global id like this: `gid = row * width + col = 3*4+2 = 14`. This global thread id maps each 2d thread to a unique 1d offset into the linear buffer, mirroring how a 1d launch would behave. 

In a 1d launch, you must flatten or unflatten coordinates manually. Mapping between 1d and 2d spaces requires division and modulus operations, which complicates the program. 
```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x; 
int row = gid / width; 
int col = tid % height;
```
In contrast, a 2d launch exposes both `x` and `y` spatial dimensions natively via `threadIdx.{x,y}` and `blockIdx.{x,y}`, making indexing and memory access patterns clearer. The two approaches are functionally equivalent, but a 2d launch makes accessing data in a kernel far more straightforward when the problem is 2d. 

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

To find out how fast a matrix multiplication kernel can be on your GPU, you can use the `cuBLAS` library, which contains highly optimized kernels written by Nvidia. These kernels are fine tuned to extract the maximum performance from the hardware, and it's extremely difficult to outperform a `cuBLAS` kernel.

All of the examples going forward will be multiplying two 4096x4096 matrices in single precision (SGEMM). 

Performance is measured in TFLOPS (trillion floating point operations per second). In order to calculate the theoretical maximum FP32 performance for my GPU (a 5070 Ti): 
- 70 SMs * 128 Cores per SM = 8960 Cuda Cores
- Each Cuda core performs 2 operations per clock cycle (FMA = fused multiply-add) 
- Boost clock: 2.45 GHz  = 2.45 * 10^9 cycles per second
- Equals approximately 44 TFLOPS. 

Now, let's estimate the number of operations required to multiply two square matrices of size 4096. Each of the `N^2` entries in matrix C requires a dot product between a row of A and a column of B, consisting of `N` multiplications and `N` additions. That’s `2N` floating point operations per entry, yielding a total of `2N^3` FLOPs.

So the total number of operations is `2*4096^3 = 137,438,953,472`. 

GFLOPS = `(2*4096^3) / (execution time in seconds × 10^9)`

The `cuBLAS` kernel hovers around 34 TFLOPS on my GPU. 
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

For each output in C, we calculate the [dot product](https://en.wikipedia.org/wiki/Dot_product) of row `i` from matrix A and column `j` from matrix B. This is an incredibly slow way to multiply matrices: the time complexity is `O(n^3)` and it only achieves around 19 gflops for a 1024x1024 matrix. This example is missing SIMD instructions, use of multiple cores, cache-friendly access for B (memory access not coalesced), to name a few. 

Numpy delegates matrix multiplication to high performance BLAS libraries, which use multithreading and SIMD. They're extremely optimized, and a good way to figure out how far you are from the theoretical performance of your hardware. 
```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python -c "import numpy as np, time; N=4096; A=np.random.rand(N,N).astype(np.float32); B=np.random.rand(N,N).astype(np.float32); t0=time.time(); C=A@B; t1=time.time(); flops=2*N**3; dt=t1-t0; print(f'Time: {dt:.4f} s, GFLOPS: {flops/dt/1e9:.2f}')"
```
This gets around 300 gflops on my Ryzen 7 9700x. For multi-threaded performance, just remove the environment variables (1,400 gflops).
### The simplest GPU matrix multiplication 
This code is a copy of the CPU matrix multiplication code, with the outer loops taken out. With one thread per element in the output matrix C (4096 * 4096 = 16,777,216) and 256 threads per block, we require `total_threads / 256 = 65,536` blocks. 
The launch configuration is two dimensional: 
`blockDim = (16,16)`  -> 256 threads per block,
`gridDim = (256,256)`  -> 65,536 total blocks.
```cpp
__global__ void matmul_B_row_strided(const float *a, const float *b, float *c) {
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (uint i = 0; i < 4096; ++i)
        sum += a[row * 4096 + i] * b[col * 4096 + i];
    c[row * 4096 + col] = sum;
}
```

This gets around 680 gflops on a 5070 ti. However, there is a massive problem with the way global memory is accessed in this kernel. 
#### Profiling kernels
`ncu` is a really good way to profile kernels and understand how well the kernel is utilizing the GPU. You can see SM throughput, number of cycles, DRAM bandwidth, and a lot of other important statistics. Profiling this kernel using `ncu` (most values are excluded for brevity; a full comparison table will come later): 

| Metric                     | Value (matmul_B_row_strided) | Unit   | Insight                             |
| -------------------------- | ---------------------------- | ------ | ----------------------------------- |
| Compute (SM) Throughput    | 21.79                        | %      | very low compute utilization        |
| Memory Throughput          | 98.06                        | %      | memory system saturated             |
| DRAM Throughput            | 7.52                         | %      | poor global memory efficiency       |
| Achieved Occupancy         | 99.74                        | %      | full warp scheduling                |
| Achieved Active Warps/SM   | 47.87                        | warp   | maxed out warp allocation           |
| Elapsed Cycles             | 5.63281e+08                  | cycles | long execution time                 |
| Average SM Active Cycles   | 5.62532e+08                  | cycles | SMs active throughout               |
| Average L1 Active Cycles   | 5.62532e+08                  | cycles | L1 busy entire time                 |
| Average L2 Active Cycles   | 5.04343e+08                  | cycles | high latency propagation            |
| Average DRAM Active Cycles | 2.54577e+08                  | cycles | global memory is a major bottleneck |
By looking at the elapsed cycles, you can tell that the SMs were active nearly the entire time, but throughput was only 22%. Also, the memory system (L1, L2, DRAM) was running for an alarmingly high percentage of the total cycles. We're doing almost no work in this kernel (22% SM throughput), but the cycle count is very high. This leads us to the conclusion that this kernel is memory bound.
### Naive matmul (coalesced) 
The root cause of the memory issues in the previous kernel is this line: 
```cpp
sum += a[row * 4096 + i] * b[col * 4096 + i];
```

The way we access A is row-major. This is inherently good for memory coalescing, since you're reading strictly left to right. All the values are next to each other. 

B, on the other hand access columns in a row-major array, which is very bad for coalescing.  
```
thread 0 → b[0 * 4096 + i] → b[i]
thread 1 → b[1 * 4096 + i] → b[4096 + i]
thread 2 → b[2 * 4096 + i] → b[8192 + i]
...
```
Each thread accesses a float 4096 elements away from the adjacent thread. 

The fix is simple. If you transpose B, the rows and columns are switched. Then, we can access the columns of B the same way we access the rows of A. Now, both A and B are accessed row-wise. 
```cpp
 sum += a[row * 4096 + i] * b[i * 4096 + col];
```

Below is the table for the coalesced memory accesses for the new kernel starting at the very first thread.

| threadIdx.x | col | Access `A[0][0]` (same for all) | Access `B[0][col]` | Address in B |
| ----------- | --- | ------------------------------- | ------------------ | ------------ |
| 0           | 0   | `a[0 * 4096 + 0]` = `a[0]`      | `b[0 * 4096 + 0]`  | `b[0]`       |
| 1           | 1   | `a[0]`                          | `b[0 * 4096 + 1]`  | `b[1]`       |
| 2           | 2   | `a[0]`                          | `b[0 * 4096 + 2]`  | `b[2]`       |
| ...         | ... | `a[0]`                          | ...                | ...          |
| 15          | 15  | `a[0]`                          | `b[0 * 4096 + 15]` | `b[15]`      |

With this change, the kernel achieves around 2700 gflops. 
### NCU comparison table
Here is a table comparing profiling results from the two kernels. The first column is the new kernel where we fixed the way B is accessed, and the second column is the old uncoalesced kernel that we profiled earlier.

| Metric                     | Average (matmul_B_col_contiguous) | Average (matmul_B_row_strided) | Unit  |
| -------------------------- | --------------------------------- | ------------------------------ | ----- |
| Average DRAM Active Cycles | 2.68328e+08                       | 2.54577e+08                    | cycle |
| Average L1 Active Cycles   | 1.32893e+08                       | 5.62532e+08                    | cycle |
| Average L2 Active Cycles   | 1.18857e+08                       | 5.04343e+08                    | cycle |
| Average SM Active Cycles   | 1.32893e+08                       | 5.62532e+08                    | cycle |
| Compute (SM) Throughput    | 92.33                             | 21.79                          | %     |
| DRAM Throughput            | 33.58                             | 7.52                           | %     |
| Duration                   | 57.95                             | 245.92                         | ms    |
| Elapsed Cycles             | 1.32984e+08                       | 5.63281e+08                    | cycle |
| L1/TEX Cache Throughput    | 92.37                             | 98.17                          | %     |
| L2 Cache Throughput        | 28.38                             | 9.01                           | %     |
| Memory Throughput          | 92.33                             | 98.06                          | %     |
| SM Active Cycles           | 1.32893e+08                       | 5.62532e+08                    | cycle |
| Total DRAM Elapsed Cycles  | 6.39275e+09                       | 2.70747e+10                    | cycle |
| Total L1 Elapsed Cycles    | 9.30642e+09                       | 3.94249e+10                    | cycle |
| Total L2 Elapsed Cycles    | 2.85843e+09                       | 1.21065e+10                    | cycle |
| Total SM Elapsed Cycles    | 9.30642e+09                       | 3.94249e+10                    | cycle |
| Total SMSP Elapsed Cycles  | 3.72257e+10                       | 1.577e+11                      | cycle |
| Waves Per                  | 156.04                            | 156.04                         | SM    |
The key differences to note here are: 
- SM throughout has gone up to 92%. This means we're actually doing ALU operations for most of the execution time instead of waiting for memory access. 
- Memory system cycles have gone down almost 5x
- L2 cache throughput has gone up. Because we made our memory access more linear and predictable, the GPU is able to utilize the cache more. 
- This kernel is nearly 4x faster than our previous one because of the above factors. 
### Tiled matmul 
The next step is to reduce our global memory loads even further. We can do this by using shared memory (on L1 cache, local to each block). Instead of reading from global memory each time, we can copy a tile of the matrix (usually 16x16) into shared memory, and then calculate the dot products. This increases 

### Optimizations & how can we reach peak performance? 



## Reference 

This was partially inspired by some George Hotz streams I watched:
- [how do GPUs work?](https://youtu.be/OUzm06YaUsI) 
- [can you multiply a matrix?](https://youtu.be/VgSQ1GOC86s)). 

All the code in this post can be found on [GitHub](https://github.com/boopdotpng/cuda-matmuls).

For another perspective on CUDA and GPU architecture, see the guide at [modal.com](https://modal.com/gpu-glossary). 

[Nvidia blackwell architecture whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture).

[High Yield: 5090 deep dive](https://youtu.be/rCwgAGG2sZQ)

