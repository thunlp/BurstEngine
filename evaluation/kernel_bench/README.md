#BurstAttention performance benchmarking


## Requirements


  - `flash-attention` from `opt_bwd` branch, which can be found in this project's root directory.
  - `nvcr.io/nvidia/pytorch:24.07-py3` docker image.
  - `TransformerEngine 2.1`

## Evaluation Settings

 - `hidden_size` = 5120
 - `num_attn_heads` = 40
 - `batch size` = 1
 - `seq_len` = 131072, 262144, 524288, 1048576


## Evaluation results


### Multi node (32 GPUs 4nodes 8 GPUs each)
- Transformer Engine(P2P) i.e. RingAttention+FlashAttention
`./logs/benchmark_dpa-2025_04_14_19_56_37.log`
`./logs/benchmark_dpa-2025_04_14_19_56_37.log`
| Dtype | QKV Format |  CUDA Kernel   |  B   | Seq Len |  TFLOPs/s |
|-------|------------|----------------|------|---------|-----------|
|  bf16 |   bshd     | FlashAttention |  1   |  131072  |   53.10   |
|  bf16 |   bshd     | FlashAttention |  1   |  262144  |   104.59 |
|  bf16 |   bshd     | FlashAttention |  1   | 524288  |   x  |
|  bf16 |   bshd     | FlashAttention |  1   |  1048576  |   x |

- USP

`./logs/bench_usp-2025_04_14_18_42_13.log`
| Dtype | QKV Format |  CUDA Kernel   |  B   | Seq Len |  TFLOPs/s |
|-------|------------|----------------|------|----------|----------|
|  bf16 |   bshd     | FlashAttention |  1   |  131072  |  140.41 |
|  bf16 |   bshd     | FlashAttention |  1   |  262144  |  166.84 |
|  bf16 |   bshd     | FlashAttention |  1   | 524288   |  174.46 |
|  bf16 |   bshd     | FlashAttention |  1   |  1048576 |  181.70 |

- LoongTrain DoubleRing implementation

`./logs/bench_loong-2025_04_14_17_20_44.log`
`./logs/bench_loong-2025_04_14_20_01_32.log`
| Dtype | QKV Format |  CUDA Kernel   |  B   | Seq Len |  TFLOPs/s |
|-------|------------|----------------|------|---------|-----------|
|  bf16 |   bshd     | FlashAttention |  1   |  32768  |   99.11  |
|  bf16 |   bshd     | FlashAttention |  1   |  65536  |   145.39 |
|  bf16 |   bshd     | FlashAttention |  1   | 131072  |   160.25 |
|  bf16 |   bshd     | FlashAttention |  1   |  1048576  | 140.69 |

- BurstAttention 

`./logs/benchmark_burst-2025_04_14_19_14_39.log`
| Dtype | QKV Format |  CUDA Kernel   |  B   | Seq Len |  TFLOPs/s |
|-------|------------|----------------|------|---------|-----------|
|  bf16 |   bshd     | FlashAttention |  1   |  131072  | 128.76 |
|  bf16 |   bshd     | FlashAttention |  1   |  262144  | 163.28 |
|  bf16 |   bshd     | FlashAttention |  1   | 524288  |  178.56|
|  bf16 |   bshd     | FlashAttention |  1   |  1048576  |187.74 |

