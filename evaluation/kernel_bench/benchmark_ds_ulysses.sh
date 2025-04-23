#!/bin/bash

if [ -z "$1" ]; then
  host="localhost"
else
  host=$1
fi
if [ -z "$2" ]; then
  node_size=1
else
  node_size=$2
fi

# --seq-lens 1048576 2097152 4194304\
torchrun --rdzv_backend=c10d --rdzv_id=1 --nnodes $node_size --rdzv_endpoint=$host:7778 --nproc_per_node=8 benchmark_ds_ulysses.py \
    --dtypes bf16 \
    --batch-sizes 1 \
    --seq-lens 131072 262144 524288 1048576 \
    --num-heads 32 \
    --head-dim 128 \
    --attn-mask-type causal \
    --warmup 2 \
    --iterations 4

