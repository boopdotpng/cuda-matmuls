This post is an in depth overview on GPU architecture and how to write performant GPU code. 

To write fast GPU code, you need to know how your instructions are scheduled and executed, and how memory is accessed in each kernel. Unlike a CPU, a GPU is massively parallel and requires a completely different mental model, which we will build in the following sections.
## Compute layout & memory
The fundamental hardware unit in a GPU is the SM (Streaming Multiprocessor). Here is where a SM sits inside a GPU. 
```
GPU  
├── GPC (graphics processing cluster)
│   └── TPC (texture processing cluster)
│       └── SM (streaming multiprocessor) — 70 total on RTX 5070 Ti
│           ├── 128 CUDA cores (scalar FP32/INT32 ALUs)
│           ├── 4 Tensor cores (matrix-multiply units)
│           ├── Special Function Units (exp, sin, etc.)
│           ├── Warp schedulers (typically 4 per SM)
│           ├── Register file (64–256 KB per SM)
│           ├── Load/Store units
│           └── Shared memory / L1 cache (64–164 KB)
├── L2 cache (shared across all SMs, 70 MB on 4090)
└── Global memory (DRAM, 8–192 GB depending on model)
```

### Types of memory on a GPU
Before we go through the execution model, it's important to understand where each layer of memory is on a GPU and the cost of accessing each layer. The execution model relies heavily on the memory layout. 

Memory access is the dominant bottleneck in most kernels, and performance depends not just on what you compute, but where your data lives and how it's accessed. Let’s examine the memory system in detail.

| Memory type          | Latency (cycles) | Bandwidth  | notes                        |
| -------------------- | ---------------- | ---------- | ---------------------------- |
| Global (GDDR or HBM) | 400-800          | 0.8-1 TB/s | high latency, off chip dram  |
| L2 cache             | 100-200          | 1-2 TB/s   | shared between SMs           |
| L1 cache/shared      | 20-40            | 1-2 TB/s   | per-sm, fast for threads     |
| Register file        | 1-4              | >10 TB/s   | per-core, extremely fast<br> |
#### Global memory 
Global memory is accessible to all SMs and represents the largest but slowest memory region on the GPU, typically implemented as off-chip GDDR6 or GDDR7 DRAM. It serves as the main interface between CPU and GPU, storing model weights, input datasets, and output buffers.

When you call `cudaMalloc`, the pointer returned points to a region in this memory.
#### L2 cache
L2 cache is a unified cache shared by all SMs that sits between global memory and cores. It caches reads from global memory to reduce latency and memory traffic. All global reads and writes pass through it, though reads benefit way more. 
#### L1 cache / shared memory 
L1 cache is fast, low latency cache local to each SM and typically shares physical space with shared memory. Shared memory is explicitly allocated in a kernel using the `__shared__` keyword inside a kernel and is accessible only to threads within the same block. Because of its speed and block level scope, it's often used to stage data from global memory, reducing costly global reads. This is a critical tool for optimizing memory-bound kernels.
#### Register file
These closely resemble registers on a CPU, except there are many times more per core in a GPU. Total capacity per SM is around 128 KB (architecture-dependent), which is divided across all active threads. Each thread gets its own private set of registers, and the number used per thread directly affects how many threads can run concurrently on an SM. There are different kinds of registers (general-purpose, predicate, special), but these distinctions only become relevant when examining PTX (NVIDIA’s intermediate representation) or SASS (its assembly-level counterpart).

You can view register usage by compiling with `nvcc --ptxas-options=-v kernel.cu`. This prints how many registers each thread in your kernel uses. The main constraint is that you cannot overuse registers: if your kernel allocates too many per thread, fewer threads can be scheduled on each SM, reducing occupancy. This limits the GPU's ability to hide latency by switching between warps. If usage exceeds hardware limits, the compiler will spill values into local memory, which is actually backed by slow global memory.

#### Memory coalescing 
Think of every memory access as an independent transaction. In a warp (group of 32 threads), coalescing allows the GPU to combine many smaller memory accesses into one large one. For example, let's say you had 32 floats stored in global memory. You launch 32 threads (one warp) that each read one float and perform an operation on it. Because all 32 floats in memory are stored side by side, the GPU can compress your 32 individual read instructions into one giant 1024-bit read. This reduces the number of read requests, reduces latency, and greatly speeds up your kernel. 

On the other hand, let's now hypothetically say that your 32 floats are located far apart in memory. Thread 0 reads `A[0]`, thread 1 reads `A[1000]`, etc. Now, the GPU can't merge your memory transactions. It has to issue 32 separate memory transactions and wait for the data to be available to each warp. The memory bus is now underutilized, and so is your SM, because it's spending most of the clock cycles waiting for memory accesses. Since each global memory read takes 400-800 cycles, this adds a significant amount of execution time. L1 and L2 cache partially mitigate this issue, but they only absorb repeated access to the same data. When access patterns are irregular and non-repeating, cache can't help, and memory coalescing becomes the dominant performance factor. 

### Execution model 

CUDA cores are simple ALUs that execute FP32 or INT32 operations,  one instruction per data element. Under ideal conditions (when instructions are ready and memory access isn't stalled), a core can perform one floating point operation per clock cycle. This is the basis for theoretical GPU throughput: multiplying the number of cores by the GPU's clock speed gives the peak number of instructions per second.

For a RTX 5070ti: 
- SMs: 70
- Cuda cores per SM: 128
- Total CUDA cores: 8,960
- Boost clock: 2.45 Ghz
- Peak FP32 throughput: 8,960 × 2.45 × 10⁹ × 2 = 43.9 TFLOPS (trillion floating point operations per second)

**Why are we multiplying by 2?**
Modern GPUs implement FP32 arithmetic using FMA (fused multiply-add) units, which perform a multiplication and addition per clock cycle. Since this counts as two operations, the effective throughput per core per clock cycle is doubled. 

## How is a kernel executed on a GPU? 

Real world performance, however, is usually much slower than this theoretical maximum. To figure out why, we have to go through the execution model: 

A thread is the smallest unit of execution. Every thread runs your kernel function, and all operations are scheduled on CUDA cores. 

A warp is a group 32 threads (all in the same block) that are executed at the same time. 

A block is a group of threads. Nvidia imposes a hard limit on the maximum amount of threads you can have in a block (almost always 1024). 

A grid is a grouping of blocks.




## A basic GPU kernel launch 
A kernel launch consists of the following: 
1. A GPU kernel function
2. Number of blocks 
3. Number of threads per block

A block is a group of threads that run together on one SM with access to shared memory and thread-level sync (will get to this in a future kernel). Blocks are distributed to SMs over time. If you launch more blocks than SMs, it queues the remaining blocks for future execution. A block is always fully contained in an SM. However, a SM might hold multiple blocks depending on the following: 
- Max threads per SM (limit set by Nvidia: usually 1024) 
- Max blocks per SM (32) 
- Shared memory per SM
- Register usage per thread / block (you can see this using `nvcc`)

It's important to realize that warps, not blocks, are the unit of scheduling. A SM executes one warp per cycle per warp scheduler (usually 4 per SM). An SM can run multiple warps, potentially from multiple blocks concurrently. 
#### Implicit kernel parameters 
 Consider the following kernel that adds two arrays `A+B` and stores the output in another array `C`. We'll assume that `len(a) = len(b) = len(c) = 1000`.
 
```cpp
__global__ void add(const float *a, const float *b, float *c) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	c[gid] = a[gid] + b[gid];
}
```

The `gid` calculation here is done based on a few parameters that are implicitly passed into every kernel. They tell the thread where it is in the grid relative to other threads. This is how we each thread knows which index of `c` it's updating.

```
grid (1d)
│
├── block 0 -- blockIdx.x = 0 
│   ├── thread 0 -- gid = 0 
│   ├── thread 1
│   ├── thread 2 -- threadIdx.x = 2
│
├── block 1 -- blockIdx.x = 1
│   ├── thread 0 -- gid = 3
│   ├── thread 1 -- threadIdx = 1
│   ├── thread 2
│
├── block 2 
│   ├── thread 0 -- gid = 6
│   ├── thread 1 -- gid = 7
│   ├── thread 2 -- gid = 8 

Dispatching 3 blocks with 3 threads each. (9 threads total)
```

In this kernel, the first thread will calculate `c[0] = a[0] + b[0]`, the second will calculate `c[1] = a[1] + b[1]`, and so on. 

One small caveat is that it's suboptimal to launch a non-power-of-two number of threads per block. This is due to the GPU architecture we discussed above: 
- You want warp aligned compute (threads are run in batches of 32)
- Memory access is a aligned and predictable (this is important for cache) 
 
 Generally, you decide the number of threads per block first, and that decides the number of blocks you launch. In this case, we chose 128 threads per block, meaning that we'd need 8 blocks to fully cover the 1000 element array.

| Parameter | Notes                               | Example for add kernel |
| --------- | ----------------------------------- | ---------------------- |
| blockIdx  | Which block is this thread in?      | 0 to 7                 |
| blockDim  | How many threads are in each block? | 128                    |
| threadIdx | Where in the block is this thread?  | 0 to 127               |
| gridDim   | How many total blocks are there?    | 8                      |
Looking back at the gid calculation: 
```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```
We get the global id by multiplying which block the thread is in by how many total blocks there are, and then adding the position of the current thread in the block. This gives us the global position of the thread, relative to every other thread in the dispatch. 

All the parameters here are actually three dimensional, but for this example only one dimension (x) is necessary. 2d and 3d dispatches are just abstractions over a 1d dispatch, and doesn't make a performance difference. It's mostly there to make indexing more convenient when you're indexing into matrices, as we will see later. 

**A note about overlaunching:**
You may have noticed we launched too many threads in the last example. Sometimes it's unavoidable, due to the power of 2 rule we imposed for `blockDim` earlier. The last 24 threads would end up accessing invalid memory `c[1000+]`. In order to guard against this, we place if statement in the kernel that exits if the gid is out of range.
#### Visualizing a 2d dispatch
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
```
block (0,0)         block (1,0)
| 0,0 | 0,1 |        | 0,2 | 0,3 |
| 1,0 | 1,1 |        | 1,2 | 1,3 |

block (0,1)         block (1,1)
| 2,0 | 2,1 |        | 2,2 | 2,3 |
| 3,0 | 3,1 |        | 3,2 | 3,3 |

block (0,2)         block (1,2)
| 4,0 | 4,1 |        | 4,2 | 4,3 |
| 5,0 | 5,1 |        | 5,2 | 5,3 |
```

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
### GPU Memory  

## Matrix multiplication on CPU
Matrix multiplication is one of the most common dense compute kernels in machine learning. This section covers a series of increasingly optimized method for square matrices. There's a George Hotz clip involving matrices, cubes, and some vague hand movements out there somewhere, but I think [this](http://matrixmultiplication.xyz/) is the best way to visualize it. The simplest implementation (multiplying two `N*N` matrices) goes like this: 
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

Each of the `N^2` entries in matrix C requires a dot product between a row of A and a column of B, consisting of `N` multiplications and `N` additions. That’s `2N` floating point operations per entry, yielding a total of `2N^3` FLOPs.

So the total number of operations is `2*4096^3 = 137,438,953,472`. 

GFLOPS = `(2N^3) / (execution time in seconds × 10^9)`

Numpy delegates matrix multiplication to high performance BLAS libraries, which use multithreading and SIMD. They're extremely optimized, and a good way to figure out how far you are from the theoretical performance of your hardware. 
```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python -c "import numpy as np, time; N=4096; A=np.random.rand(N,N).astype(np.float32); B=np.random.rand(N,N).astype(np.float32); t0=time.time(); C=A@B; t1=time.time(); flops=2*N**3; dt=t1-t0; print(f'Time: {dt:.4f} s, GFLOPS: {flops/dt/1e9:.2f}')"
```
This gets around 300 gflops on my Ryzen 7 9700x. For multi-threaded performance, just remove the environment variables (1,400 gflops).
## The simplest GPU matrix multiplication 
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
### Profiling kernels
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
## Naive matmul (coalesced) 
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
#### NCU comparison table
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
## Tiled matmul 
The next step is to reduce our global memory loads even further. We can do this by using shared memory (on L1 cache, local to each block). Instead of reading from global memory each time, we can copy a tile of the matrix (usually 16x16) into shared memory, and then calculate the dot products. This increases 

## Optimizations & how can we reach peak performance? 



# Resources 

This was partially inspired by some George Hotz streams I watched:
- [how do GPUs work?](https://youtu.be/OUzm06YaUsI) 
- [can you multiply a matrix?](https://youtu.be/VgSQ1GOC86s)). 

All the code in this post can be found on [GitHub](https://github.com/boopdotpng/cuda-matmuls).

For another perspective on CUDA and GPU architecture, see the guide at [modal.com](https://modal.com/gpu-glossary). 

[Nvidia blackwell architecture whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture).

