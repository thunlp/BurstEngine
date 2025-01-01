#!/bin/bash
llama_path="./llama-7b-mega-ds-zero3"
universal_llama_path="./llama-7b-mega-ds-universal"
HF_LLAMA_PATH=/data/public/opensource_models/meta-llama/Llama-2-7b-hf
ds_path=/home/hanxv/workspace/Megatron-DeepSpeed/llama-7b-mega-ds-zero3-8gpus
cmd="python "script/ds_to_universal.py" \
    --input_folder ${llama_path}/global_step0 \
    --num_extract_workers 1 \
    --no_strict \
    --num_merge_workers 1 \
   --output_folder ${llama_path}/global_step0_universal"
echo $cmd
$cmd
