#!/bin/bash

# 8 GPU training script

# Set up environment variables
export MASTER_ADDR="g48"
export MASTER_PORT="29500"
# export WORLD_SIZE=8

# Run the training script with distributed data parallel
torchrun --nproc_per_node=8 --nnodes=2   --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT test_burst_ckpt.py --backend torch

