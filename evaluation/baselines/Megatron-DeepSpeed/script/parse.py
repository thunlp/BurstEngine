import re
from collections import defaultdict


def parse_bash_command(command):
    # Regular expression to match key-value pairs and flags
    pattern = re.compile(r'--([^= ]+)(?:[ =]([^ ]+))?')

    # Dictionary to store the parsed parameters
    params = defaultdict(list)

    # Split the command by spaces while keeping quoted strings together
    tokens = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', command)

    i = 0
    while i < len(tokens):
        match = pattern.match(tokens[i])
        if match:
            key, value = match.groups()
            # Handle flags without values (e.g., --deepspeed)
            if value is None and (i + 1 < len(tokens)) and not tokens[i + 1].startswith('--'):
                value = tokens[i + 1]
                i += 1
            params[key].append(value if value is not None else True)
        i += 1

    # Convert single-value lists to values
    for key in params:
        if len(params[key]) == 1:
            params[key] = params[key][0]

    return dict(params)
cmd1 = "deepspeed --num_gpus=8 /home/hanxv/workspace/Megatron-DeepSpeed/script/../pretrain_gpt.py --override-opt_param-scheduler --adam-beta1 0.9 --adam-beta2 0.95 --tensor-model-parallel-size 1 --ds-sequence-parallel-size 1 --init-method-std 0.006 --lr-decay-tokens 300000000000 --lr-warmup-tokens 3000000000 --micro-batch-size 1 --exit-duration-in-mins 30000000 --global-batch-size 8 --disable-bias-linear --num-layers 32 --hidden-size 4096 --num-attention-heads 32 --seq-length 4096 --max-position-embeddings 4096 --train-tokens 300000000000 --train-samples 146484375 --lr 2e-5 --min-lr 1.0e-6 --zero-reduce-scatter --lr-decay-style cosine --split 949,50,1 --log-interval 10 --eval-interval 100 --eval-iters 10 --swiglu --ffn-hidden-size 11008 --use-rotary-position-embeddings --normalization rmsnorm --no-query-key-layer-scaling --weight-decay 0.1 --clip-grad 1.0 --hysteresis 2 --zero-reduce-scatter --num-workers 0 --fp16 --normalization rmsnorm --seed 1234 --finetune --no-async-tensor-model-parallel-allreduce --tensorboard-queue-size 1 --log-timers-to-tensorboard --log-batch-size-to-tensorboard --log-validation-ppl-to-tensorboard --tensorboard-dir output/tensorboard/gpt_8B_tok300B_lr2e-5_min1.0e-6_w3000M_d300B_cosine_gbs8_mbs1_g1_z3_seed1234_rebase_g1002_2024.07.29_10.19.55 --swiglu --ffn-hidden-size 11008 --checkpoint-activations --log-optimizer-states-to-tensorboard --data-path data/codeparrot_content_document --tokenizer-type HFTokenizer --tokenizer-model /data/public/opensource_models/meta-llama/Llama-2-7b-hf --data-impl mmap --deepspeed --deepspeed_config ../ckpt/ds_config_gbs8_mbs1_log10_zero3.json --zero-stage 3 --pipeline-model-parallel-size 1 --no-pipeline-parallel --deepspeed-activation-checkpointing 2>&1 | tee output/log//gpt_8B_tok300B_lr2e-5_min1.0e-6_w3000M_d300B_cosine_gbs8_mbs1_g1_z3_seed1234_rebase_g1002_2024.07.29_10.19.55.log"
cmd2 = "deepspeed --num_gpus=8 ./pretrain_gpt.py --finetune --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --override-opt_param-scheduler --adam-beta1 0.9 --adam-beta2 0.95 --checkpoint-activations --lr-warmup-iters 0 --weight-decay 0.1 --clip-grad 1 --num-layers 32 --untie-embeddings-and-output-weights --hidden-size 4096 --zero-reduce-scatter --num-attention-heads 32 --tensorboard-dir output/tensorboard/llama7b_1_8_true_hpz_1_zero_3_bs8_seq1024_seed_rebase_dmx-login03_2024.07.29_18.06.18 --log-timers-to-tensorboard --log-batch-size-to-tensorboard --log-validation-ppl-to-tensorboard --ffn-hidden-size 11008 --attention-dropout 0 --hidden-dropout 0 --no-query-key-layer-scaling --disable-bias-linear --normalization rmsnorm --use-rotary-position-embeddings --swiglu --seq-length 1024 --max-position-embeddings 1024 --micro-batch-size 1 --global-batch-size 8 --train-iters 3500 --lr 5e-6 --min-lr 1e-7 --lr-decay-iters 320000 --lr-decay-style cosine --use-flash-attn-v2 --log-interval 10 --hysteresis 2 --tensorboard-queue-size 1 --eval-iters 100 --eval-interval 100 --data-path /home/hanxv/workspace/Megatron-DeepSpeed/script/data/codeparrot_content_document --split 100,0,0 --fp16 --zero-stage 3 --tokenizer-type HFTokenizer --tokenizer-model /data/public/opensource_models/meta-llama/Llama-2-7b-hf --deepspeed_config ./script/ds_finetune_config_true.json --deepspeed --deepspeed-activation-checkpointing --distributed-backend nccl --num-workers 0 --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion --no-gradient-accumulation-fusion --repeated-dataloader --no-pipeline-parallel --load ./ckpt/llama-7b-mega-ds-zero3-8gpus-fp16 --make-vocab-size-divisible-by 256"
dict1 = parse_bash_command(cmd1)
dict2 = parse_bash_command(cmd2)
for key in dict1:
    if key not in dict2:
        print("key not in dict2: ", key)
    else:
        if dict1[key] != dict2[key]:
            print("diff value: key: ", key, "value_origin: ", dict1[key], "value_finetune: ", dict2[key])

for key in dict2:
    if key not in dict1:
        print("key not in dict1: ", key)
from IPython import embed;embed()

