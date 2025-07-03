nnodes=1
gpus=8
tp=8
sp=burst
# model=13b
# model=70b
if [ -z $WORLD_SIZE ]; then
  WORLD_SIZE=1
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
model=7b
bs=1
export TOK_PATH=/llama-7b/
bash pre.sh
cmd="torchrun --nnodes=$WORLD_SIZE --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:7778 pretrain_llama_hug_tokenizer.py --model-config config/${model}/config.json --vocab config/70b/vocab.txt --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_dock.json --start-step 1 --offload --batch-size ${bs} --max-length $seqlen --tp ${CP_SIZE} --sp ${method} --flash cuda --spzero --tokenizer-path $TOK_PATH --ckpt  2>&1 |tee  ${nnodes}nnodes_${gpus}gpus_${tp}tp_${method}sp.log"
echo $cmd
eval $cmd
