## Result of AWS Network 

- A cluster of 2 AWS p3.2xlarge instances;
- The results are from the recv side (TC can only control recv side.)
- The data size is determined by the bytes of a practical macro-batch (for GPT-3 XL).
  - macro-batch size: 4, 8;
  - sequence length: 2048;
  - model size: 2048.

| Network setting                     | Gloo-64 MB | NCCL-64 MB | Gloo-128 MB | NCCL-128 MB |
|-------------------------------------|------------|------------|-------------|-------------|
| default (about 0.1ms; up to 10Gbps) | 107 ms     | 55 ms      | 216 ms      | 112 ms      |
| delay 1ms                           | 110 ms     | 59 ms      | 222 ms      | 115 ms      |
| delay 5ms                           | 274 ms     | 80 ms      | 512 ms      | 134 ms      |
| delay 10ms                          | 532 ms     | 113 ms     | 972 ms      | 171 ms      |
| bandwidth 5Gbps                     | 110 ms     | 109 ms     | 218 ms      | 217 ms      |
| bandwidth 2Gbps                     | 271 ms     | 270 ms     | 542 ms      | 541 ms      |
| bandwidth 1Gbps                     | 541 ms     | 541 ms     | 1082 ms     | 1082 ms     |
| delay 1ms  bandwidth 5Gbps          | 115 ms     | 110 ms     | 224 ms      | 220 ms      |
| delay 5ms  bandwidth 2Gbps          | 311 ms     | 275 ms     | 581 ms      | 542 ms      |
| delay 10ms  bandwidth 1Gbps         | 620 ms     | 543 ms     | 1160 ms     | 1083 ms     |
