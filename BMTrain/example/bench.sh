# export NCCL_DEBUG=INFO
export MASTER_ADDR=bjdx1
export MASTER_PORT=4080
DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes 2
    --rdzv_id=1 
    --rdzv_backend=c10d 
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
)
cmd="torchrun ${DISTRIBUTED_ARGS[@]} benchmark.py"
$cmd
