#!/bin/bash
endpoint=g4005
cmd="export NCCL_P2P_DISABLE=1;torchrun --nnodes=1 --nproc_per_node=8  --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$endpoint pretrain_llama_hug_tokenizer.py --model-config config/7b/config.json --vocab config/7b/vocab.txt --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_datasets_laptop.json --start-step 1 --offload --batch-size 1 --max-length 4096 --tp 8 --flash cuda --sp burst --spzero"
echo $cmd
eval $cmd
