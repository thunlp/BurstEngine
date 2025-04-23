#!/bin/bash

if [ -z "$1" ]; then
  host="localhost"
else
  host=$1
fi
if [ -z "$2" ]; then
  world_size=1
else
  world_size=$2
fi
export NCCL_IB_QPS_PER_CONNECTION=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO
    # --seq-lens 1048576 2097152 4194304  \
torchrun --rdzv_backend=c10d --rdzv_id=1 --nnodes $world_size --rdzv_endpoint=$host:7778 --nproc_per_node=8 benchmark_burst_signle_entry.py \
        --dtypes bf16 \
        --seq-lens 131072 262144 524288 \
        --batch-sizes 1 \
        --num-heads 40 \
        --head-dim 128 \
        --causal \
        --double-ring true \
        --opt-bwd true \
        --use-striped false \
        --warmup 5 \
        --iterations 10 \
        --profile \
