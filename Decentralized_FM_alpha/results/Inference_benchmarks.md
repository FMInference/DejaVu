# Throughput Test of Inference Tasks

## GPT-J

- Test on 4 P3.2xlarge;
- Configure:
  - Input sequence length: 512
  - Generate sequence length: 32
  - Max batch size: 48;
- Tuning token level pipe micro batch size:

| Token micro-batch-size | Prompt time | Generate time | Overall time |
|------------------------|-------------|---------------|--------------|
| 48                     | 1.93 s      | 4.89 s        | 6.82 s       | 
| 24                     | 1.92 s      | 3.34 s        | 5.26 s       | 
| 12                     | 1.93 s      | 2.59 s        | 4.52 s       | 
| 6                      | 1.93 s      | 4.95 s        | 6.88 s       | 
| 3                      | 1.92 s      | 9.21 s        | 11.13 s      | 
| 1                      | 1.94 s      | 29.55s        | 31.48 s      | 


## GPT-66B

- Test on 1 P4d.24xlarge (8 40G A100)
- Configure
  - Input sequence length: 1024
  - Generate sequence length: 100
  - Max batch size: 20;
  - 8 layer per GPU;
- Tuning token level pipe micro batch size:


| Token micro-batch-size | Prompt time | Generate time | Overall time | FLOPS per GPU |
|------------------------|-------------|---------------|--------------|---------------|
| 20                     | 2.71 s      | 42.78 s       | 45.49 s      | 16.44 TFLOPS  |
| 10                     | 2.72 s      | 42.82 s       | 45.54 s      | 16.42 TFLOPS  |
| 4                      | 2.71 s      | 130.78 s      | 133.49 s     | 5.60 TFLOPS   |


## GPT-175B (Estimated run)

- Test 1/3 of the workflow on 1 P4d.24xlarge (8 40G A100); cannot get 3 machines at the same time.
- Configure
  - Input sequence length: 1024
  - Generate sequence length: 100
  - Max batch size: 20;
  - 4 layer per GPU;
- Tuning token level pipe micro batch size:


| Token micro-batch-size | Prompt time | Generate time  | Overall time | Estimate time on 3 nodes | FLOPS per GPU |
|------------------------|-------------|----------------|--------------|--------------------------|---------------|
| 20                     | 2.14 s      | 29.58 s        | 31.72 s      | 95.16 s                  | 6.94 TFLOPS   |
| 10                     | 2.14 s      | 25.89 s        | 28.03 s      | 84.09 s                  | 7.86 TFLOPS   |
| 4                      | 2.14 s      | 66.34 s        | 68.48  s     | 205.44 s                 | 3.22 TFLOPS   |


## GPT-175B A40/A100-80G Run

- Tested on two machine on FluidStack, each has
  - 8 A40 GPU: 48GB RAM; 149 TFLOPS
  - Total FLOPS of the cluster: 2.387 PFLOPS
  - Notice the number reported in the deepspeed blog is about 19.2 token/s in 16 A100
  - Results:

| Prompt Length | Token Generation Length | Batch Size | Batch Time | Token Throughput | Total FPLOPS  | Cluster Efficiency |
|---------------|-------------------------|------------|------------|------------------|---------------|--------------------|
| 512           | 50                      | 56         | 42.3 s     | 66.2 token/s     | 226.35 TFLOPS | 9.4%               |
| 1900          | 100                     | 18         | 95.1 s     | 19.0 token/s     | 135.25 TFLOPS | 5.6%               |


- Tested on one machine on FluidStack that has:
  - 8 A100 GPU: 80 GB RAM, 312 TFLOPS
  - Total FLOPS of the cluster: 2.496 PFLOPS
  - Results:
  
| Prompt Length | Token Generation Length | Batch size | Batch Time | Token Throughput | Total FPLOPS  | Cluster Efficiency |
|---------------|-------------------------|------------|------------|------------------|---------------|--------------------|
| 512           | 50                      | 42         | 20.9 s     | 100.4 token/s    | 395.88 TFLOPS | 15.9%              |
| 1900          | 100                     | 14         | 42.7 s     | 32.8 token/s     | 234.19 TFLOPS | 9.3%               |



## GPT-175B CPU Token Generation

- Tested on a single m6i.32xlarge with 128 vCPU 512 GB RAM. ($6.1/hour)
- Configure
  - 96 Layers in bfloat16; Intel_extension_for_pytorch enabled
  - ~~Removed batchNorm (not supported in CPU, this need to be fixed for real run.)~~
  - Use torch.cpu.amp.autocast_mode() for batchNorm. 

- Setting 1.
  - Input sequence length: 1024
  - Generate sequence length: 100
  - FLOPs for a token:  0.35 TFLOPs
  - FLOPs for a seq prompt: 361.19 TFLOPs

| Batch size | Prompt time | Prompt FLOPS | Generate time per token per batch | Token throughput | Token FLOPS  |
|------------|-------------|--------------|-----------------------------------|------------------|--------------|
| 1          | 449.27 s    | 0.80         | 5.19 s                            | 0.19             | 0.067 TFLOPS |
| 4          | 1431.55 s   | 1.01         | 11.76 s                           | 0.34             | 0.11 TFLOPS  |
| 8          | -           | -            | 17.18 s                           | 0.45             | 0.16 TFLOPS  |
| 16         | -           | -            | 24.37 s                           | 0.65             | 0.22 TFLOPS  |
| 20         | -           | -            | 28.57 s                           | 0.70             | 0.25 TFLOPS  |
| 24         | -           | -            | OOM                               |                  |              |


- Setting 2.
  - Input sequence length: 512
  - Generate sequence length: 50
  - FLOPs for a token:  179.35 TFLOPs
  - FLOPs for a seq prompt:  0.35 TFLOPs

| Batch size | Prompt time  | Prompt FLOPS | Generate time per token per batch | Token throughput | Token FLOPS |
|------------|--------------|--------------|-----------------------------------|------------------|-------------|
| 1          | -            | -            | 2.30 s                            | 0.43             | 0.15        |
| 4          | -            | -            | 8.44 s                            | 0.47             | 0.16        |      
| 8          | -            | -            | 10.73 s                           | 0.74             | 0.26        |            
| 16         | -            | -            | 14.77 s                           | 1.08             | 0.38        |            
| 32         | -            | -            | 24.95 s                           | 1.28             | 0.49        |            
| 40         | -            | -            | 26.09 s                           | 1.53             | 0.54        |
| 48         | -            | -            | OOM                               |                  |             |


## 175B CPU-GPU Hybrid

- Basic Setting
  - Prompt: 24 g5.2xlarge A10g 24GB. ($1.21/hour)
  - Token: m6i.32xlarge with 128 vCPU 512 GB RAM. ($6.1/hour)

- Setting 1:
  - 24 g5.2xlarge
  - 30 m6i.32xlarge; 
  - Prompt length 512, token length: 50;
  - Per token computation: 0.35 TFLOP

| Micro-Batch size | Per CPU Token throughput | Cluster Throughput | CPU FLOPS per Node |
|------------------|--------------------------|--------------------|--------------------|
| 4                | 0.82                     | 24.6               | 0.287              |
| 8                | 1.01                     | 30.3               | 0.354              |
| 16               | -                        | -                  | -                  |


- Setting 2:
  - 24 g5.2xlarge
  - 30 m6i.32xlarge; 
  - Prompt length 1900, token length: 100;
  - Per token computation: 0.35 TFLOP

| Micro-Batch size | Per CPU Token throughput | Cluster Throughput | CPU FLOPS per Node |
|------------------|--------------------------|--------------------|--------------------|
| 1                | 0.14                     | 4.2                | 0.05               |
