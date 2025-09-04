
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
convert=0
cd ../
vocab_path="`pwd`/script/data/vocab.json"
merge_path="`pwd`/script/data/merges.txt"
# data_path="./ckpt/alpaca_data.json"
data_path="`pwd`/script/data/codeparrot_content_document"
export CUDA_MAX_CONNECTIONS=1
# weights link: https://huggingface.co/huggyllama/llama-7b
HF_LLAMA_PATH=/data/public/opensource_models/meta-llama/Llama-2-7b-hf

MICRO_BATCH_SIZE=1
gpus=8
nnodes=32
GLOBAL_BATCH_SIZE=$(( MICRO_BATCH_SIZE * nnodes* gpus ))
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
TP=1
PP=1
pretrain=1
sp_size=1
# require to align with weight dimensions
HIDDEN_SIZE=16384
FFN_HIDDEN_SIZE=53248
NUM_LAYERS=300
NUM_HEADS=128
NUM_KV_HEADS=16
SEQ_LENGTH=1024
######################################

llama_path="./ckpt/llama-7b-mega-ds-zero3-8gpus-fp16"
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
QUANTIZE=true
DS_CONFIG=./ckpt/ds_finetune_config_${QUANTIZE}.json
HPZ_SIZE=1
ZERO_STAGE=3
# "zero_hpz_partition_size": $HPZ_SIZE,
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 100,
  "zero_optimization": {
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true,
        "ratio": 0.8

    },
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
    "enabled": true,
    "loss_scale": 128,
    "loss_scale_window": 10,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 20
  },
  "wall_clock_breakdown": true
}
EOT

convert_args="torchrun --nnodes=1 --nproc_per_node=8 tools/hf2megads_weight_converter_ds.py --hf-ckpt-num-shards 2 --origin-hf-ckpt-dir $HF_LLAMA_PATH --save $MEGA_DS_LLAMA_PATH"
if [ $pretrain -eq 1 ] && [ $convert -eq 0 ]; then
    echo "Pretrain"
    filename="pretrain_gpt.py"
    data_path="`pwd`/script/data/codeparrot_content_document"
else
    echo "Fine-tuning"
    filename="finetune_llama.py"
    data_path="`pwd`/script/alpaca_data.json"
fi
# finetune_args="deepspeed --num_gpus=${gpus}  ./${filename} \
# --finetune"
if [ $nnodes -eq 1 ]; then
  master_addr="localhost"
else
  if [ -z $1 ]; then
    master_addr="g31"
  else
    echo "master_addr=$1"
    master_addr=$1
  fi 
fi
finetune_args="torchrun --nproc-per-node=${gpus} --nnodes ${nnodes} --master-addr=${master_addr} --master-port=12306 --rdzv-id=4 --rdzv-backend=c10d --rdzv-endpoint=${master_addr}:12306 ./${filename} \
--finetune"

# --untie-embeddings-and-output-weights \
jobname="9g1015b_${nnodes}_${gpus}_${QUANTIZE}_hpz_${HPZ_SIZE}_zero_${ZERO_STAGE}_bs${GLOBAL_BATCH_SIZE}_seq${SEQ_LENGTH}"
jobname="${jobname}_seed${seed}_rebase"fine
host="${HOSTNAME}"
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
username=$(whoami)
output_home="output"
log_path="${output_home}/log/"
checkpoint_path="${output_home}/checkpoint/${jobname}"
tensorboard_dir="${output_home}/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_dir}

# --untie-embeddings-and-output-weights \
# --load ${llama_path} \
# --tokenizer-model $HF_LLAMA_PATH \
# --tokenizer-type HFTokenizer \
comm_args="--tensor-model-parallel-size $TP \
--pipeline-model-parallel-size $PP \
--override-opt_param-scheduler \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--checkpoint-activations \
--lr-warmup-iters 0 \
--weight-decay 0.1 \
--clip-grad 1 \
--num-layers $NUM_LAYERS \
--untie-embeddings-and-output-weights \
--hidden-size $HIDDEN_SIZE \
--zero-reduce-scatter \
--cpu-optimizer \
--num-attention-heads $NUM_HEADS \
--num-key-value-heads $NUM_KV_HEADS \
--tensorboard-dir ${tensorboard_path} \
--log-timers-to-tensorboard \
--log-batch-size-to-tensorboard \
--log-validation-ppl-to-tensorboard \
--ffn-hidden-size $FFN_HIDDEN_SIZE \
--attention-dropout 0 \
--hidden-dropout 0 \
--no-query-key-layer-scaling \
--disable-bias-linear \
--normalization rmsnorm \
--use-rotary-position-embeddings \
--swiglu \
--seq-length $SEQ_LENGTH \
--max-position-embeddings $SEQ_LENGTH \
--no-masked-softmax-fusion \
--no-bias-gelu-fusion \
--ds-sequence-parallel-size ${sp_size} \
--no-bias-dropout-fusion \
--no-gradient-accumulation-fusion \
--micro-batch-size $MICRO_BATCH_SIZE \
--global-batch-size $GLOBAL_BATCH_SIZE \
--train-iters 3500 \
--lr 5e-6 \
--min-lr 1e-7 \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--use-flash-attn-v2 \
--log-interval 1 \
--hysteresis 2 \
--tensorboard-queue-size 1 \
--eval-iters 100 \
--eval-interval 100 \
--data-path $data_path \
--split 100,0,0 \
--fp16 \
--zero-stage 3 \
--tokenizer-type GPT2BPETokenizer \
--vocab-file $vocab_path \
--merge-file $merge_path \
--deepspeed_config $DS_CONFIG \
--deepspeed \
--checkpoint-activations \
--deepspeed-activation-checkpointing \
--distributed-backend nccl \
--num-workers 0 \
--repeated-dataloader \
--no-pipeline-parallel \
--make-vocab-size-divisible-by 256"
if [ $convert -eq 1 ]; then
    echo "Converting weights"
    task_args="$convert_args"
else
    task_args="$finetune_args"
fi

echo $task_args
full_cmd="$task_args $comm_args | tee ${log_path}/${jobname}_${host}_${current_time}.log"
# echo $full_cmd
eval $full_cmd
# eval "$full_cmd"


