#!/bin/bash
dir=`pwd`
###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
seq_len=1024
multi_nodes=0
HF_LLAMA_PATH=/data/public/opensource_models/meta-llama/Llama-2-7b-hf
load_llama_path="../llama-7b-mega-ds-zero3/"
load_llama_path="../llama-7b-mega-ds-zero3-8gpus-fp16/"
save_llama_path="../llama-7b-pure/"



## The "GPT-3 XXX" below are configs from GPT-3 paper
## https://arxiv.org/abs/2005.14165, choose based on
## your desired model size or build your own configs

## init_std is standard deviation for weight initialization. Usually larger
## model needs lower std. We used a heuristic equation of sqrt(1/3/hidden_size)
## from the MT-NLG 530B work (https://arxiv.org/pdf/2201.11990.pdf)

## We changed min_lr to a lower number (1.0e-6), which we found is able to
## provide better zero-shot eval results.

## GPT-3 Small 125M
# model_size=0.125
# num_layers=12
# hidden_size=768
# num_attn_heads=12
# global_batch_size=256
# lr=6.0e-4
# min_lr=1.0e-6
# init_std=0.02

## GPT-3 Medium 350M
# model_size=0.35
# num_layers=24
# hidden_size=1024
# num_attn_heads=16
# global_batch_size=256
# lr=3.0e-4
# min_lr=1.0e-6
# init_std=0.018

## GPT-3 Large 760M
# model_size=0.76
# num_layers=24
# hidden_size=1536
# num_attn_heads=16
# global_batch_size=256
# lr=2.5e-4
# min_lr=1.0e-6
# init_std=0.015

## GPT-3 XL 1.3B
# model_size=1.3
# num_layers=24
# hidden_size=2048
# num_attn_heads=16
# global_batch_size=8
# lr=2.0e-4
# min_lr=1.0e-6
# init_std=0.013

## GPT-3 2.7B
# model_size=2.7
# num_layers=32
# hidden_size=2560
# num_attn_heads=32
# global_batch_size=512
# lr=1.6e-4
# min_lr=1.0e-6
# init_std=0.011

## GPT-3 6.7B
# model_size=6.7
# num_layers=32
# hidden_size=4096
# num_attn_heads=32
# global_batch_size=8
# lr=1.2e-4
# min_lr=1.0e-6
# init_std=0.009

## GPT-3 13B
model_size=13
num_layers=40
hidden_size=5120
num_attn_heads=40
global_batch_size=8
lr=1.0e-4
min_lr=1.0e-6
init_std=0.008

## GPT-3 18B
model_size=18
num_layers=40
hidden_size=6144
num_attn_heads=32
global_batch_size=24
lr=1.0e-5
min_lr=1.0e-6
init_std=0.008

## CPM-9G-8B
model_size=8
num_layers=32
hidden_size=4096
num_attn_heads=32
global_batch_size=8
lr=2e-5
min_lr=1.0e-6
init_std=0.006
act="silu"
dim_ff=11008
# vocab_size=119696
quantize="false"
if [ "${quantize}" = "true" ]; then
    echo "Using quantization"
    hpz_partition_size=8
else
    echo "Do not use quantization"
    hpz_partition_size=1
fi

## GPT-3 175B
# model_size=175
# num_layers=96
# hidden_size=12288
# num_attn_heads=96
# global_batch_size=1536
# lr=0.6e-4
# min_lr=1.0e-6
# init_std=0.005
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens.
train_tokens_in_billion=300
train_tokens=$((${train_tokens_in_billion} * 1000000000))

## train_samples is another termination condition and also affect the number of 
## data samples to be indexed. Since we want to reach the train_tokens
## above, and data efficiency techniques may change num tokens in some samples,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by train_samples.
train_samples=$(( 300 * 1000000000 * 2 / ${seq_len} ))

## Another wall-clock time termination condition in minutes. Set it large
## enough to avoid undesired early termination.
exit_duration=30000000
###############################################################################
### lr configs
## lr warmup and decay duration.
## Original GPT-3 paper uses 375M warmup tokens and 260B cosine decay tokens.
## Here we increase the warmup tokens to 3B since when batch size warmup is not
## used, there are more tokens per step. Thus we need to increase warmup tokens
## to make sure there are enough warmup steps, which is important for training
## stability.
lr_warmup_tokens_in_million=0
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))
## Here we changed the LR decay tokens to align with total train tokens, since
## related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
## learning rate schedule to match the number of training tokens results in the
## best final model quality 
lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000000000))
lr_decay_style="cosine"
###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
## Currently we only support MP=1 with SP>1
mp_size=2

## Sequence parallelism, 1 is no SP
sp_size=1

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Note that currently both curriculum learning and random-LTD are NOT
## compatible with pipeline parallelism.
pp_size=2
no_pp="false"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
zero_stage=1

## Total number of GPUs. ds_ssh is from DeepSpeed library.
if [[ $multi_nodes -eq 1 ]]; then
    num_gpus=$(($(ds_ssh -f ./hostfile "nvidia-smi --query-gpu=name --format=csv,noheader" | wc -l)-2))
else
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node=$(( ${num_gpus} / ${num_gpus_pernode} ))

## Data parallel size.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} / ${sp_size} ))

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Reduce it manually if GPU OOM
# batch_size=$(( ${global_batch_size} / ${dp_size} ))
batch_size=1

###############################################################################
### Misc configs
log_interval=2
eval_iters=10
eval_interval=100
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=100
estimated_train_iter=$((${train_tokens} / ${seq_len} / ${global_batch_size}))
# save_interval=$((${estimated_train_iter} / ${num_save}))
save_interval=100

## Activation checkpointing saves GPU memory, but reduces training speed
activation_checkpoint="true"
# activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"
seed=1234
num_workers=0

vocab_path="data/vocab.json"
merge_path="data/merges.txt"


data_path="$dir/data/codeparrot_content_document"

prescale_grad="false"
jobname="cpm9G_${model_size}B_tok${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}_min${min_lr}_w${lr_warmup_tokens_in_million}M_d${lr_decay_tokens_in_billion}B_${lr_decay_style}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_g${num_gpus}"
if [[ $zero_stage -gt 0 ]]; then
    jobname="${jobname}_z${zero_stage}"
    prescale_grad="false"
fi
if [[ $sp_size -gt 1 ]]; then
    jobname="${jobname}_sp${sp_size}"config
fi
if [[ $mp_size -gt 1 ]]; then
    jobname="${jobname}_mp${mp_size}"
fi
if [ "${no_pp}" = "false" ]; then
    jobname="${jobname}_pp${pp_size}"
fi
if [ "${quantize}" = "true" ]; then
    jobname="${jobname}_quantize"
fi
jobname="${jobname}_seed${seed}_rebase"

username=$(whoami)
output_home="output"
log_path="${output_home}/log/"
checkpoint_path="${output_home}/checkpoint/${jobname}"
tensorboard_dir="${output_home}/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
###############################################################################
data_options=" \
    --data-path ${data_path} \
    --data-impl mmap"

## If CL is used, make sure to set "--split" the same as what you used during
## offline data analysis&indexing.
megatron_options=" \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size 1 \
    --ds-sequence-parallel-size ${sp_size} \
    --init-method-std ${init_std} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-tokens ${lr_warmup_tokens} \
    --micro-batch-size ${batch_size} \
    --exit-duration-in-mins ${exit_duration} \
    --global-batch-size ${global_batch_size} \
    --zero-reduce-scatter \
    --disable-bias-linear \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-tokens ${train_tokens} \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $vocab_path \
    --merge-file $merge_path \
    --vocab-size 32000 \
    --train-samples ${train_samples} \
    --lr ${lr} \
    --finetune \
    --min-lr ${min_lr} \
  --no-masked-softmax-fusion \
  --no-bias-gelu-fusion \
  --no-bias-dropout-fusion \
  --no-gradient-accumulation-fusion \
    --lr-decay-style ${lr_decay_style} \
    --split 98,2,0 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --no-query-key-layer-scaling \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --weight-decay 0.1 \
    --untie-embeddings-and-output-weights \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers ${num_workers} \
    --fp16 \
    --use-rotary-position-embeddings \
    --normalization rmsnorm \
    --no-query-key-layer-scaling \
    --seed ${seed} \
    --no-async-tensor-model-parallel-allreduce \
    --use-flash-attn-v2 \
    --tensorboard-queue-size 1 \
    --log-batch-size-to-tensorboard \
    --timing-log-level 0 \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${tensorboard_path}"

if [ "${act}" = "silu" ]; then
    echo "Using SiLU activation"
    megatron_options="${megatron_options} \
    --swiglu \
    --ffn-hidden-size ${dim_ff}"
fi

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

config_json="ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}_quantize${quantize}.json"
template_json="./ds_config_gpt_TEMPLATE.json"
echo "Using quantize ${quantize}"
sed "s/GBSIZE/${global_batch_size}/" ${template_json} \
    | sed "s/MBSIZE/${batch_size}/" \
    | sed "s/QUANTIZE/${quantize}/" \
    | sed "s/HPZ_SIZE/${hpz_partition_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/${prescale_grad}/" \
      > ${config_json}

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}"

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0
for (( node = 0; node <= num_node-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
        local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
        iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} :  ${iteration} ))
    fi
done
if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    ds_ssh "echo $iteration > $iteration_file"
    ds_ssh "echo $iteration_2 > $iteration_file_2"
fi
if [[ $num_node -gt 1 ]]; then
  hf="--hostfile ./hostfile"
else
  hf=""
fi
MASTER_ADDR=g3028
MASTER_ADDR=localhost
MASTER_PORT=12306
dist_opt="--rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
cmd="torchrun --nnodes=1 --nproc_per_node=8 ${dist_opt} ${dir}/../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}"
# cmd="torchrun --nnodes=1 --nproc_per_node=8 ${dist_opt} ${dir}/../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} 2>&1 | tee ${log_path}/${jobname}_${host}_${current_time}.log"
#
echo $cmd
eval '$cmd 2>&1 | tee ${log_path}/${jobname}_${host}_${current_time}.log'
