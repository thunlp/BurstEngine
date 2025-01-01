#!/bin/bash
llama_path=./"llama-7b-mega-ds"
HF_LLAMA_PATH=/data/public/opensource_models/meta-llama/Llama-2-7b-hf

cmd="python ../tools/hf2megads_weight_converter.py \
--hf-ckpt-num-shards 2 \
--origin-hf-ckpt-dir $HF_LLAMA_PATH \
--save $llama_path"

echo $cmd
$cmd
