
nnodes=1
gpus=8
# method=burst_sparse
CP_SIZE=8
# model=13b
# model=70b
#
export LOG_FILE=/BurstEngine/apps/llama/7b_flex.log
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

# 标准输出重定向命令
if [ -z $MODEL ]; then
  model=7b
else
  model=$MODEL
fi
bs=1
echo $model
touch ${WORLD_SIZE}nnodes_${gpus}gpus_${CP_SIZE}tp_${method}sp_${seqlen}seqlen_${model}.log
export TOK_PATH=/llama-7b/
cmd="torchrun --nnodes=$WORLD_SIZE --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:7778 pretrain_llama_hug_tokenizer.py  --model-config config/${model}/config.json  --vocab config/70b/vocab.txt --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_dock.json --start-step 1 --bf16 --batch-size ${bs} --max-length $seqlen --tp ${CP_SIZE} --sp ${method} --offload --flash cuda --spzero --tokenizer-path $TOK_PATH --ckpt  2>&1 |tee  ${WORLD_SIZE}nnodes_${gpus}gpus_${CP_SIZE}tp_${method}sp_${seqlen}seqlen_${model}.log"
echo $cmd
eval $cmd
