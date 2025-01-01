## SpeedUp

### TFlops(MFU) Plot

#### Multi-Node

4x figure
| GPUS       | Model       | Seqlen   | Methods               |
|------------|-------------|----------|-----------------------|
| 32xGPUs    | 7b/13b model| 128k-512k| burst_double, burst_ulysses, ulysses, megatron-cp, burst_sparse |
| 64xGPUs    | 30b/70b model| 128k-512k| burst_double, burst_ulysses, ulysses, megatron-cp, burst_sparse |

- TODO 
    - 32xGPUs
        - [ ] 13b 128k-1024k all methods
        - [ ] 7b 128-1024k all methods
            - [ ] now only burst_sparse left
            - [ ] others

    - Best SP Splits
        different model/GPUS/seqlen, the smallest SP split

#### Single-Node

2xfigure
| GPUS | Model | Seqlen | Methods |
|------|-------|--------|---------|
| 8xGPUs | 7b/13b model | 32k-256k | burst, ulysses, megatron-cp, burst_sparse |


### Throughput Table

### Scale Plot(Speed Up / Megatron-cp)
GPUs: 1x 2x 4x 8x 
Model: 7b model
Statics: TFlops/Throughput
SeqLen: 32k

GPUs: 8x 16x 32x 64x
Model: 13b model
Statics: TFlops/Throughput
SeqLen: 128k

GPUs: 1x 2x 4x 8x 16x 32x 64x
Model: 0.7b  1.3b  2.7b  7b  13b  30b  70b model
Statics: TFlops/Throughput
SeqLen: 128k

### Ablaition Study



