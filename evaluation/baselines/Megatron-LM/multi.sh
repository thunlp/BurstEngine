#!/bin/bash
MAX_RESTARTS=0
export NCCL_IB_QPS_PER_CONNECTION=8
export CUDA_DEVICE_MAX_CONNECTIONS=1


DATA_PATH="/data/"
DATASET="codeparrot_content_document"
GPUS_PER_NODE=8
method_str=""
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

if [ -z $TP_SIZE ]; then
  TP_SIZE=1
fi

if [ -z $MASTER_ADDR ]; then
  MASTER_ADDR=localhost
  MASTER_PORT=6238
  single=true
fi

if [ "$method" == "burst" ]; then
  echo "USE BURST"
  method_str="--use-burst"
  method_str+=" --create-dq-comm"
elif [ "$method" == "burst_double" ]; then
  echo "USE BURST DOUBLE"
  method_str="--use-burst"
  method_str+=" --double-ring"
  method_str+=" --create-dq-comm"
elif [ "$method" == "burst_ulysses" ]; then
  echo "USE BURST ULYSSES"
  method_str="--use-burst"
  method_str+=" --use-ulysses"
elif [ "$method" == "ulysses" ]; then
  echo "USE ULYSSES"
  method_str="--use-ulysses"
elif [ "$method" == "megatron-cp" ]; then
  echo "USE MEGATRON-CP"
  method_str="--context-parallel-size ${CP_SIZE}"
fi
echo "$method_str"

DISTRIBUTED_ARGS="--max_restarts 1 --nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs
MICRO_BATCH=1
GLOBAL_BATCH=$((GPUS_PER_NODE * WORLD_SIZE / CP_SIZE / TP_SIZE * MICRO_BATCH))
PP_SIZE=1
SEQ_LEN=$seqlen
#MODEL CONFIG
if [ -z $MODEL ]; then
  MODEL="7b"
fi

if [ "$MODEL" == "13b" ]; then
  NUM_LAYERS=40
  HIDDEN_SIZE=5120
  FFN_HIDDEN_SIZE=13824
  KV_HEADS=40
  NUM_ATTN_HEADS=40
elif [ "$MODEL" == "30b" ]; then
  NUM_LAYERS=64
  HIDDEN_SIZE=6144
  FFN_HIDDEN_SIZE=$((HIDDEN_SIZE * 4))
  KV_HEADS=64
  NUM_ATTN_HEADS=64
elif [ "$MODEL" == "2.7b" ]; then
  NUM_LAYERS=32
  HIDDEN_SIZE=2560
  FFN_HIDDEN_SIZE=$((HIDDEN_SIZE * 4))
  KV_HEADS=32
  NUM_ATTN_HEADS=32
elif [ "$MODEL" == "7b" ]; then
  NUM_LAYERS=32
  HIDDEN_SIZE=4096
  FFN_HIDDEN_SIZE=$((HIDDEN_SIZE * 4))
  KV_HEADS=32
  NUM_ATTN_HEADS=32
fi
sp_method=$method


tensorboard_dir="./output/tensorboard/model_${MODEL}_seqlen_${SEQ_LEN}_gbs_${GLOBAL_BATCH}_mbs_${MICRO_BATCH}_${DATETIME}_tp_${TP_SIZE}_sp_${sp_method}"
echo tensorboard_dir: $tensorboard_dir
mkdir -p $tensorboard_dir
options=" \
	--tensor-model-parallel-size $TP_SIZE \
	--tensorboard-dir $tensorboard_dir \
	--log-timers-to-tensorboard \
	--timing-log-level 2 \
  --method $method \
	--pipeline-model-parallel-size $PP_SIZE \
        --num-layers ${NUM_LAYERS} \
        --no-create-attention-mask-in-dataloader \
        --hidden-size ${HIDDEN_SIZE} \
      --attention-dropout 0 \
      --hidden-dropout 0 \
      --use-rotary-position-embeddings \
      --normalization RMSNorm \
      --disable-bias-linear \
      --num-attention-heads ${NUM_ATTN_HEADS} \
      --seq-length ${SEQ_LEN} \
      --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
      --max-position-embeddings ${SEQ_LEN} \
	--micro-batch-size ${MICRO_BATCH} \
	--global-batch-size ${GLOBAL_BATCH} \
        --lr 6.0e-5 \
	--min-lr 6.0e-6 \
        --lr-decay-style cosine \
		--train-iters 10 \
        --log-interval 1 \
        --eval-iters 0 \
        --sequence-parallel
        --eval-interval 1000 \
	--data-path ${DATA_PATH}/${DATASET} \
	--vocab-file ${DATA_PATH}/vocab.json \
	--merge-file ${DATA_PATH}/merges.txt \
	--initial-loss-scale 65536 \
	--save-interval 1000 \
	--split 98,2,0 \
  --use-distributed-optimizer \
	--clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
    --swiglu \
	--init-method-std 0.006 \
  --bf16 \
	--position-embedding-type rope \
  --transformer-impl transformer_engine \
  --log-throughput \
  --context-parallel-size ${CP_SIZE} \
		--use-flash-attn \
    --timing-log-level 2 \
      --causal \
    --log-file $LOG_FILE \
    --causal-tflops \
	  --recompute-granularity full \
	  --recompute-method uniform \
	  --recompute-num-layers 1 \
	"

  # --tp-comm-overlap \

run_cmd="torchrun $DISTRIBUTED_ARGS ${DIR}/pretrain_gpt.py ${options} ${method_str}"
echo $run_cmd
$run_cmd 
# bash /home/test/test01/sa/kill.sh

set +x

