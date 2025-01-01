nnodes=1
gpus=8
sp=burst
# model=13b
# model=70b
#
nsys_cmd="nsys profile -s none -t nvtx,cuda -o ./burst_comm.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi  --capture-range-end=stop"

export LOG_FILE=/workspace/workspace/burst_exp/apps/llama/13b_burst4nodes_1024k.log
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_MIN_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=4
if [ -z $WORLD_SIZE ]; then
  WORLD_SIZE=1
else
  bash pre.sh
  # pass
fi

if [ -z $method ]; then
  method=$2
fi

if [ -z $seqlen ]; then
  seqlen=$1
fi

if [ -z $CP_SIZE ]; then
  CP_SIZE=$3
fi

if [ -z $MASTER_ADDR ]; then
  MASTER_ADDR=localhost
  single=true
fi
#!/bin/bash
if [ -z $ABLATION ]; then 
  ab_args=""
else
  ab_args="--ablation $ABLATION "
fi 
# 标准输出重定向命令
if [ -z $MODEL ]; then
  model=13b
else
  model=$MODEL
fi
bs=1
echo $model
touch ${WORLD_SIZE}nnodes_${gpus}gpus_${CP_SIZE}tp_${method}sp_${seqlen}seqlen_${model}.log
export TOK_PATH=/workspace/workspace/models/llama-7b/
cmd="torchrun --nnodes=$WORLD_SIZE --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:7778 pretrain_llama_hug_tokenizer.py   --offload --model-config config/${model}/config.json  --vocab config/70b/vocab.txt  --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_dock.json --start-step 1 --bf16 --batch-size ${bs} --max-length $seqlen --tp ${CP_SIZE} --sp ${method} $ab_args --flash cuda --spzero --tokenizer-path $TOK_PATH --ckpt  2>&1 |tee  ${WORLD_SIZE}nnodes_${gpus}gpus_${CP_SIZE}tp_${method}sp_${seqlen}seqlen_${model}.log"
if [ "$PROFILE" = "true" ]; then
  cmd="$nsys_cmd $cmd"
fi
echo $cmd
eval $cmd
