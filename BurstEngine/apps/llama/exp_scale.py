import socket
import subprocess
import os
import argparse

def run(cmd):
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
            'tflops': 'NaN',
            'params': 'NaN',
            'toks': 'NaN',
            'std': 'NaN'
        }
    return info

def add_method_to_cmd(cmd, method):
    if method == 'ring': cmd += f" --sp ring"
    elif method.startswith('burst'): cmd += f" --sp burst"
    if 'flash' in method: cmd += " --flash triton"
    if 'zero' in method: cmd += " --spzero"
    if 'mask' in method: cmd+= " --mask"
    return cmd
if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--type", type=str, default="all", choices=["all", "train", "infer", "ppl"])
    args = args_parser.parse_args()
    inf = False
    train = False
    ppl = False
    if args.type == "all":
        print("run inf and train")
        inf = True
        train = True
        only_print_cmd = True
    elif args.type == "train":
        train = True
        print("run train")
    elif args.type == "infer":
        inf = True
        print("run infer")
    elif args.type == "ppl":
        ppl = True
        print("run ppl")

    # with open("ppl.txt", "w") as f:
    #     print("method,loss,ppl", file=f)
    #     for method in ['tp', 'tp+flash', 'ring', 'burst', 'burst+flash', 'burst+flash+zero']:
    #         cmd = f"torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 pretrain_llama_hug_tokenizer.py --model-config config/7b/config.json --vocab config/7b/vocab.txt --batch-size 1 --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_datasets_laptop.json --start-step 1 --max-length 4096 --tp 8 --offload --ppl --eval_only"
    #         cmd += " --load /home/hanxv/tpCPM-Live/llama-2-7b-hf/pytorch_model.pt"
    #         cmd = add_method_to_cmd(cmd, method)
    #         info = run(cmd)
    #         print(info)
    #         print(f"{method},{info['loss']},{info['ppl']}", file=f)
    #         f.flush()
    # endpoint = "localhost:12345"
    def print_rank(s):
        if rank == 0:
            print(s)
    if inf:
        for nodes in [1]:
            if nodes == 1:
                endpoint="localhost:12306"
                rank=0
            else:
                node_list = ['g3009','g3018','g3014','g3025']
                endpoint = "g3009:7778"
                hostname = socket.gethostname()
                rank = node_list.index(hostname)

            for bw in ['low']:
                for dim in ['7b']:
                    with open(f"{dim}-infer-scale.txt-gpu8-tpv3v1-causal.txt", "w") as f:
                        print("bw,dim,gpus,seqlen,method,mem,time,toks,params,std", file=f)
                        for gpus in [8]:
#                            if dim == "7b":
                            # seqlens = [65536]
                            #elif dim == "13b":
                            #seqlens =  [1024, 2048,4096]
                            # for seqlen in seqlens :
                            for seqlen in [4096, 8192, 16384, 32768, 65536, 131072, 262144]:
                            #  for seqlen in [131072]:
                                for method in ['tp+sp+flash','tp+flash']:
                                #for method in ['burst+flash','tp+sp+flash']:
                                    cmd = "export NCCL_P2P_DISABLE=1;" if bw == 'low' else ""
                                    # cmd += f"torchrun --nnodes={nodes} --nproc_per_node={gpus} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 pretrain_llama_hug_tokenizer.py --model-config config/{dim}/config.json --vocab config/{dim}/vocab.txt --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_datasets_laptop.json --start-step 1 --offload --eval_only"
                                    cmd += f"torchrun --nnodes={nodes} --nproc_per_node={gpus} --node_rank={rank} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint={endpoint} pretrain_llama_hug_tokenizer.py --model-config config/{dim}/config.json --vocab config/{dim}/vocab.txt --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_datasets_laptop.json --start-step 1 --offload --eval_only"
                                    cmd += f" --batch-size 1"
                                    if "tp+sp" in method:
                                        cmd += " --tp-sp"
                                    cmd += f" --max-length {seqlen}"
                                    cmd += f" --tp {gpus*nodes}"
                                    cmd = add_method_to_cmd(cmd, method)
                                    info = run(cmd)
                                    print(info)
                                    if rank == 0:
                                        print(f"{bw},{dim},{gpus},{seqlen},{method},{info['mem_infer']},{info['infer']},{info['toks']},{info['params']},{info['std']}", file=f)
                                    f.flush()
    node_list = ['g3014','g3015','g3019','g3023']
    endpoint = "g3011:7778"
    hostname = socket.gethostname()
    rank = node_list.index(hostname)
    if train:
        try:
            def print_func(*args, **kwargs):
                if rank == 0:
                    print(*args, **kwargs)
            f = None
            for nodes in [1]:
                if nodes == 1:
                    endpoint="localhost:12306"
                    rank=0
                for bw in ['low']:
                    for dim in ['7b']:
                        for gpus in [8]:
                            if rank == 0:
                                if f is not None:
                                    f.close()
                                f =  open(f"train-7b-8-gpu-tp-causal-scale.txt", "w")
                            else:
                                f = None
                            print_func("bs,bw,dim,gpus,seqlen,method,mem,time,tflops,toks,params,std", file=f)
                            for bs in [1,2,4,8]:
                                
                                #seqlens =  [4096]
                                # make bs 1 65536 retest
                                #for seqlen in [65536]:
                                # seqlens = [4096]
                                # if bs == 1:
                                seqlens = [4096]
                                #     seqlens = [12288]
                                # seqlens = [32768,65536,98304,131072]
                                # elif bs == 2:
                                #     seqlens = [12288]
                                # else:
                                #     seqlens = [12288]

                                for seqlen in seqlens:
                                    methods = ['tp+sp+flash']
                                    # if seqlen == 4096:
                                        # method = ['ring']
                                    for method in methods:
                                    #for method in ['tp+flash', 'tp+sp+flash', 'burst+flash', 'burst+flash+zero', 'ring']:
                                    # for method in ['tp+flash+mask']:
                                        cmd = "export NCCL_P2P_DISABLE=1;" if bw == 'low' else ""
                                        cmd += f"torchrun --nnodes={nodes} --nproc_per_node={gpus} --node_rank={rank} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint={endpoint} pretrain_llama_hug_tokenizer.py --model-config config/{dim}/config.json --vocab config/{dim}/vocab.txt --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_datasets_laptop.json --start-step 1 --offload"
                                        cmd += f" --batch-size {bs}"
                                        cmd += f" --max-length {seqlen}"
                                        cmd += f" --tp {gpus * nodes}"
                                        if "tp+sp" in method:
                                            cmd += " --tp-sp"
                                        cmd = add_method_to_cmd(cmd, method)
                                        info = run(cmd)
                                        print(info)
                                        if rank == 0:
                                            print(f"{bs},{bw},{dim},{gpus},{seqlen},{method},{info['mem_train']},{info['train']},{info['tflops']},{info['toks']},{info['params']},{info['std']}", file=f)
                                        if rank == 0:
                                            f.flush()
        finally:
            if rank == 0:
                f.close()

