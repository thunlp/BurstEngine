#!/bin/bash

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes 2
    --rdzv_id=1 
    --rdzv_backend=c10d 
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
)
cmd="torchrun ${DISTRIBUTED_ARGS[@]} benchmark.py"
echo $cmd
$cmd
