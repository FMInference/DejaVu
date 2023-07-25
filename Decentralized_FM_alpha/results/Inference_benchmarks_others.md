
## OPT-1.3B on p2.8xlarge (K80)

GPU: K80(11GB) * 8

Test on p2.8xlarge;

Configure:
- Input sequence length: 512
- Generate sequence length: 50

| bs=16, 512+50   | time/batch | token throughput |
|-----------------|------------|------------------|
| DeepSpeed       | 11.86      |     67.5 tokens/s         |
| Parallelformers | 10.9s      |     73.4 tokens/s         |
| Pipeline (ours) | 6.50s      |     123.1 tokens/s        |

Starting from bs=32, DeepSpeed and Parallelformers always produce OOM.

| bs=128, 512+50   | time/batch | token throughput |
|-----------------|------------|------------------|
| DeepSpeed       | OOM      |         OOM        |
| Parallelformers | OOM      |        OOM         |
| Pipeline (ours) | 26.9s    |     238 tokens/s   |
