import subprocess
def run(cmd):
    print(cmd)
    try:
        output = subprocess.run(cmd, shell=True, capture_output=True, check=True)
        output = output.stdout.decode("utf-8").strip()
        output = output.split("\n")[-1]
        info = {s.split(':')[0].strip(): float(s.split(':')[1].strip()) for s in output.split('|')[1:-1]}
    except:
        info = {
            'mem_infer': 'NaN',
            'mem_train': 'NaN',
            'infer': 'NaN',
            'train': 'NaN',
            'loss': 'NaN',
            'ppl': 'NaN',
        }
    return info
cmd = "torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=g3016:7778 pretrain_llama_hug_tokenizer.py --model-config config/7b/config.json --vocab config/7b/vocab.txt --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_datasets_laptop.json --start-step 1 --offload --eval_only --batch-size 1 --max-length 4096 --tp 8"
run(cmd)