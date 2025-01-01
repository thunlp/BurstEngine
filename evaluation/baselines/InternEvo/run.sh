
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
export CUDA_DEVICE_MAX_CONNECTIONS=1
cmd="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py --config configs/7B_llama2.py"
echo $profile
if [ "$profile" = "true" ]; then
  cmd+=" --profiling "
fi
echo $cmd
$cmd
