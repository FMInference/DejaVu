# Result of GPT-3 Small  

## Pipeline Alone

For pipeline only, we have:

- A cluster of 3 AWS p3.2xlarge instances;

- The results are from the rank-[0] node (this is a fair setting considering the bubble overhead of Gpipe)

- The current dataset makes the embedding (handled by rank-[0] node) and loss function (handled by rank-[N-1] node) very light-weighted, thus we evenly partitioned the layers over a homogenous network.
   
- GPT-3 small scale and partition:

  - number of layer: 12 (4 on each node) 
  - model dim: 768
  - number of head: 12
  - sequence length: 2048;
  - max batch dim (due to DRAM limits): 64
  - micro-batch dim: 1 
  - Storage of a micro-batch: 6 MB 
  - Storage of parameters in an instances: 108 MB
  - Storage of parameters in the first instance of pipeline (due to some NLP embedding): 198 MB

### Gpipe based pipeline parallel 

- TC run:

  - Full precision:

| Network setting                     | Micro batch size: 1 | 
|-------------------------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 5.91 s              |
| delay 1ms  bandwidth 5Gbps          | 6.01 s              | 
| delay 5ms  bandwidth 2Gbps          | 6.07 s              | 
| delay 10ms  bandwidth 1Gbps         | 7.90 s              | 
| delay 50ms  bandwidth 1Gbps         | 8.77 s              | 

  - fp16:

| Network setting                     | Micro batch size: 1 | 
|-------------------------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 2.25 s              |
| delay 1ms  bandwidth 5Gbps          | 2.26 s              | 
| delay 5ms  bandwidth 2Gbps          | 2.55 s              | 
| delay 10ms  bandwidth 1Gbps         | 4.01 s              | 
| delay 50ms  bandwidth 1Gbps         | 4.78 s              | 


- Real Run

|                      | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 4 |
|----------------------|---------------------|---------------------|---------------------|
| Oregon/Virginia/Ohio | 10.26 s             | 10.57 s             | 10.93 s             |

### 1F1B based pipeline parallel 
- Not faster than Gpipe, left for further optimization.

| Network setting                     | Micro batch size: 1 | Micro batch size: 2 | Micro batch size: 4 |
|-------------------------------------|---------------------|---------------------|---------------------|
| default (about 0.1ms; up to 10Gbps) | 6.54 s              | 6.65 s              | 6.86 s              |
| delay 1ms  bandwidth 5Gbps          | 6.86 s              | 7.05 s              | 7.22 s              |
| delay 5ms  bandwidth 2Gbps          | 8.42 s              | 8.40 s              | 8.57 s              |
| delay 10ms  bandwidth 1Gbps         | 11.21 s             | 11.03 s             | 11.35 s             |



## Pipeline + Data Parallel

- Gpipe + centralized PS Data parallel:
- Each GPipe pipeline runs a batch of size 64;
- The global batch size is 64 * DP degree:
  - DP degree 1: 64
  - DP degree 4: 256
  - DP degree 16: 1024
- fp32 (updated on 2022/02/14).

  - Centralized PS (on rank-0):

| Network setting                     | DP Degree: 1 | DP Degree: 4 | DP Degree: 16 |
|-------------------------------------|--------------|--------------|---------------|
| default (about 0.1ms; up to 10Gbps) | 5.91 s       | 6.41 s       | 6.47 s        |
| delay 1ms  bandwidth 5Gbps          | 6.01 s       | 6.67 s       | 6.74 s        |
| delay 5ms  bandwidth 2Gbps          | 6.07 s       | 7.71 s       | 7.86 s        |
| delay 10ms  bandwidth 1Gbps         | 7.90 s       | 11.01 s      | 11.17 s       |
| delay 50ms  bandwidth 1Gbps         | 8.77 s       | 12.59 s      | 12.77 s       |

- fp16 (updated on 2022/04/10).
  
- AllReduce (This is not going to be used.):
  
| Network setting                     | DP Degree: 1 | DP Degree: 4 | DP Degree: 16 |
|-------------------------------------|--------------|--------------|---------------|
| default (about 0.1ms; up to 10Gbps) | 2.25 s       | 2.53 s       | 2.82 s        |
| delay 1ms  bandwidth 5Gbps          | 2.26 s       | 3.16 s       | 4.18 s        |
| delay 5ms  bandwidth 2Gbps          | 2.55 s       | 5.59 s       | 10.06 s       |
| delay 10ms  bandwidth 1Gbps         | 4.01 s       | 10.19 s      | 19.09 s       |
| delay 50ms  bandwidth 1Gbps         | 4.78 s       | 27.08 s      | 62.86 s       |

- Centralized PS (on rank-0):
  
| Network setting                     | DP Degree: 1 | DP Degree: 4 | DP Degree: 16 |
|-------------------------------------|--------------|--------------|---------------|
| default (about 0.1ms; up to 10Gbps) | 2.25 s       | 2.44 s       | 2.49 s        |
| delay 1ms  bandwidth 5Gbps          | 2.26 s       | 2.52 s       | 2.97 s        |
| delay 5ms  bandwidth 2Gbps          | 2.55 s       | 3.19 s       | 4.09 s        |
| delay 10ms  bandwidth 1Gbps         | 4.01 s       | 5.53 s       | 5.92 s        |
| delay 50ms  bandwidth 1Gbps         | 4.78 s       | 6.23 s       | 7.80 s        |

- Sharded PS (updated on 2022/04/13):

| Network setting                     | DP Degree: 1 | DP Degree: 4 | DP Degree: 16 |
|-------------------------------------|--------------|--------------|---------------|
| default (about 0.1ms; up to 10Gbps) | 2.25 s       | s            | s             |
| delay 1ms  bandwidth 5Gbps          | 2.26 s       | s            | s             |
| delay 5ms  bandwidth 2Gbps          | 2.55 s       | s            | s             |
| delay 10ms  bandwidth 1Gbps         | 4.01 s       | s            | s             |
| delay 50ms  bandwidth 1Gbps         | 4.78 s       | s            | s             |

## ZeRO-S3 

- I used FSDP from fairscale for this benchmark:
- Due to memory limitation, I can get the batch size to 1, 2 in each instance:
- For a cluster with 12 instances:  
  - batch size 1: global batch size: 12
  - batch size 2: global batch size: 24


| Network setting                      | batch size: 1 | batch size: 2 | 
|--------------------------------------|---------------|---------------|
| default (about 0.1ms; up to 10Gbps)  | 1.02 s        | 1.26 s        | 
| delay 1ms  bandwidth 5Gbps           | 1.53 s        | 1.78 s        |
| delay 5ms  bandwidth 2Gbps           | 3.53 s        | 3.76 s        | 
| delay 10ms  bandwidth 1Gbps          | 6.84 s        | 7.07 s        |
| delay 50ms bandwidth 1Gbps           | 12.54 s       | 12.77 s       |
