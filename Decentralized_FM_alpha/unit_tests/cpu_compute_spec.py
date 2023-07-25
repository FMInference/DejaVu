import torch
import time

batch_sizes = [4, 8, 16, 32, 64, 128, 256]

print("Real workflow:")
for b in batch_sizes:
    tensor1 = torch.randn(b, 96, 1, 128)
    tensor2 = torch.randn(b, 96, 128, 512)
    start_time = time.time()
    res = torch.matmul(tensor1, tensor2)
    end_time = time.time()
    FLOPs = 2*b*96*1*128*512
    compute_time = end_time-start_time
    print(f"Batch size <{b}> Compute time: {compute_time}s, total flops: {FLOPs}, TFLOPS: {FLOPs/compute_time/1e12}")


dims = [256, 512, 1024, 2048, 4096, 8192]
print("Peak Flops:")
for d in dims:
    tensor1 = torch.randn(d, d)
    tensor2 = torch.randn(d, d)
    start_time = time.time()
    res = torch.matmul(tensor1, tensor2)
    end_time = time.time()
    FLOPs = 2*d*d*d
    compute_time = end_time-start_time
    print(f"Dim <{d}> Compute time: {compute_time}s, total flops: {FLOPs}, TFLOPS: {FLOPs/compute_time/1e12}")