export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8 
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
# export NCCL_DEBUG=INFO
export MASTER_ADDR=g48
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
