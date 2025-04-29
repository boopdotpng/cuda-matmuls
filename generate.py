import numpy as np; N=4096; 
A=np.random.rand(N,N).astype(np.float32)
B=np.random.rand(N,N).astype(np.float32).T
A.tofile("./bins/A.bin")
B.tofile("./bins/B.bin")
(A@B).tofile("./bins/C.bin")
print(f"saved A * B (both {N}x{N}) and C")