
#!/bin/bash
MAX_RESTARTS=0
export NCCL_IB_QPS_PER_CONNECTION=8
export MEGATRON_LOGGING_LEVEL=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
export CUDA_DEVICE_MAX_CONNECTIONS=1
DATA_PATH="data/"
DATASET="codeparrot_content_document"
GPUS_PER_NODE=8
WORLD_SIZE=1
MASTER_ADDR=g41
MASTER_PORT=6000
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs
MICRO_BATCH=1
GLOBAL_BATCH=1
PP_SIZE=1
TP_SIZE=1
CP_SIZE=8
SEQ_LEN=$1
#MODEL CONFIG
MODEL="7b"
NUM_LAYERS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
KV_HEADS=32
NUM_ATTN_HEADS=32
sp_method="none"

tensorboard_dir="./output/tensorboard/model_${MODEL}_seqlen_${SEQ_LEN}_gbs_${GLOBAL_BATCH}_mbs_${MICRO_BATCH}_${DATETIME}_tp_${TP_SIZE}_sp_${sp_method}"
echo tensorboard_dir: $tensorboard_dir
mkdir -p $tensorboard_dir
options=" \
	--tensor-model-parallel-size $TP_SIZE \
	--tensorboard-dir $tensorboard_dir \
	--log-timers-to-tensorboard \
	--timing-log-level 2 \
	--pipeline-model-parallel-size $PP_SIZE \
        --num-layers ${NUM_LAYERS} \
        --no-create-attention-mask-in-dataloader \
        --hidden-size ${HIDDEN_SIZE} \
      --attention-dropout 0 \
      --hidden-dropout 0 \
      --use-rotary-position-embeddings \
      --untie-embeddings-and-output-weights \
      --swiglu \
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
	--init-method-std 0.006 \
	--bf16 \
	--swiglu \
	--position-embedding-type rope \
	--untie-embeddings-and-output-weights \
  --transformer-impl transformer_engine \
  --log-throughput \
	--sequence-parallel \
  --context-parallel-size ${CP_SIZE} \
		--use-flash-attn \
    --timing-log-level 2 \
      --causal \
      --causal-tflops \
	  --recompute-granularity full \
	  --recompute-method uniform \
	  --recompute-num-layers 1 \
	"
# --use-ulysses \

      # --double-ring \
      # --create-dq-comm \
      # --use-burst \
  # --tp-comm-overlap \
	# --distribute-saved-activations \

run_cmd="torchrun $DISTRIBUTED_ARGS ${DIR}/pretrain_gpt.py ${options}"
echo $run_cmd
$run_cmd
set +x

