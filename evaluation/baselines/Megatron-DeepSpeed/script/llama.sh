cd ../
DS_CONFIG=./examples_deepspeed/finetune_hf_llama/ds_config.json
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
convert=1
vocab_path="data/vocab.json"
merge_path="data/merges.txt"


data_path=./alpaca_data.json
# weights link: https://huggingface.co/huggyllama/llama-7b
HF_LLAMA_PATH=/data/public/opensource_models/meta-llama/Llama-2-7b-hf
model_type="fp16"
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
TP=1
PP=1
# require to align with weight dimensions
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=1024
######################################
llama_path=./ckpt/"llama-7b-mega-ds-zero3-8gpus-${model_type}"
# llama_path=./"llama-7b-pure"
MEGA_DS_LLAMA_PATH=${llama_path}

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################
QUANTIZE=false
HPZ_SIZE=1
ZERO_STAGE=3
if [ $model_type == "fp16" ]; then
    is_fp16="true"
    is_bf16="false"
elif [ $model_type == "bf16" ]; then
    is_fp16="false"
    is_bf16="true"
else
    is_fp16="false"
    is_bf16="false"
fi
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 100,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "reduce_bucket_size": 10000000,
    "reduce_scatter": true,
    "zero_quantized_weights": $QUANTIZE,
    "zero_quantized_gradients": $QUANTIZE,
    "zero_hpz_partition_size": $HPZ_SIZE,
    "contiguous_gradients": true,
    "overlap_comm": true 
  },
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "fp16": {
    "enabled": ${is_fp16},
    "loss_scale": 65536,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 11
  },
   "bf16": {
    "enabled": ${is_bf16},
   }
  "wall_clock_breakdown": true
}
EOT


convert_args="torchrun --nnodes=1 --nproc_per_node=8 tools/hf2megads_weight_converter_ds.py --hf-ckpt-num-shards 2 --origin-hf-ckpt-dir $HF_LLAMA_PATH --save $MEGA_DS_LLAMA_PATH"

finetune_args="deepspeed --num_gpus=8 finetune_llama.py \
--finetune"

comm_args="--tensor-model-parallel-size $TP \
--pipeline-model-parallel-size $PP \
--lr-warmup-iters 2000 \
--weight-decay 0.1 \
--deepspeed-activation-checkpointing \
--clip-grad 1 \
--num-layers $NUM_LAYERS \
--vocab-file ${vocab_path} \
--merge-file ${merge_path} \
--hidden-size $HIDDEN_SIZE \
--num-attention-heads $NUM_HEADS \
--ffn-hidden-size $FFN_HIDDEN_SIZE \
--attention-dropout 0 \
--hidden-dropout 0 \
--no-query-key-layer-scaling \
--disable-bias-linear \
--normalization rmsnorm \
--use-rotary-position-embeddings \
--untie-embeddings-and-output-weights \
--swiglu \
--seq-length $SEQ_LENGTH \
--max-position-embeddings $SEQ_LENGTH \
--micro-batch-size $MICRO_BATCH_SIZE \
--global-batch-size $GLOBAL_BATCH_SIZE \
--train-iters 3500 \
--lr 2e-5 \
--tensorboard-dir tensorboard_output_${QUANTIZE} \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--use-flash-attn-v2 \
--log-interval 1 \
--eval-iters 100 \
--eval-interval 100 \
--data-path $data_path \
--data-impl mmap \
--save-interval 1500 \
--split 100,0,0 \
--${model_type} \
--zero-stage 3 \
--tokenizer-type HFTokenizer \
--tokenizer-model $HF_LLAMA_PATH \
--deepspeed_config ./examples_deepspeed/finetune_hf_llama/ds_config.json \
--deepspeed \
--distributed-backend nccl \
--num-workers 0 \
--no-masked-softmax-fusion \
--no-bias-gelu-fusion \
--no-bias-dropout-fusion \
--no-gradient-accumulation-fusion \
--repeated-dataloader \
--no-pipeline-parallel \
--make-vocab-size-divisible-by 256"
if [ $convert -eq 1 ]; then
    echo "Converting weights"
    task_args="$convert_args"
else
    echo "Fine-tuning"
    task_args="$finetune_args"
fi

echo $task_args
full_cmd="$task_args $comm_args"
echo $full_cmd
eval "$full_cmd"

