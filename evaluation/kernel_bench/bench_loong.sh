#!/bin/bash

if [ -z "$1" ]; then
  host="localhost"
else
  host=$1
fi
# if [ -z "$2" ]; then
#   world_size=1
# else
#   world_size=$2
# fi
bash pre.sh
export INTRA_SIZE=1
echo $host
# export NCCL_DEBUG=INFO
# export CUDA_DEVICE_MAX_CONNECTIONS=1
    # c-seq-lens 1048576 2097152 4194304
torchrun --rdzv_backend=c10d --rdzv_id=1 --nnodes $2 --rdzv_endpoint=$host:7778 --nproc_per_node=8 benchmark_doublering.py \
    --dtypes bf16 \
    --qkv-formats bshd \
    --batch-sizes 1 \
    --seq-lens 1048576 \
    --num-heads 40 \
    --head-dim 128 \
    --gqa-groups 40 \
    --attn-mask-type causal \
    --context-parallel True \
    --warmup 10 \
    --iterations 10 \
    --use-ulysses False \
    --head-first False \
    --profile \


