import numpy as np; N=4096; 
A=np.random.rand(N,N).astype(np.float32)
B=np.random.rand(N,N).astype(np.float32)
A.tofile("./bins/A.bin")
B.tofile("./bins/B.bin")
B.T.tofile("./bins/B_t.bin")
(A@B).tofile("./bins/C.bin")
print(f"saved A * B (both {N}x{N}) and C")
