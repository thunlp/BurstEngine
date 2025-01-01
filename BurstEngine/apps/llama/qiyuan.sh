nnodes=1
gpus=8
tp=8
sp=burst
model=10b
bs=5
# bash qiyuan.sh $HOST $seqlen
export TOK_PATH=/data/public/opensource_models/meta-llama/Llama-2-7b-hf/
cmd="torchrun --nnodes=$nnodes --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$1:7778 pretrain_llama_hug_tokenizer.py --model-config config/10b/config.json --vocab config/70b/vocab.txt --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_datasets_laptop.json --start-step 1 --offload --batch-size ${bs} --max-length $2 --tp ${tp} --sp ${sp} --flash cuda --spzero --tokenizer-path $TOK_PATH --ckpt  2>&1 |tee  ${nnodes}nnodes_${gpus}gpus_${tp}tp_${sp}sp.log"
echo $cmd
eval $cmd
