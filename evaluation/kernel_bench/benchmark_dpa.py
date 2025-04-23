#!/usr/bin/env python3

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
from contextlib import nullcontext
import numpy as np
from transformer_engine.pytorch.attention import DotProductAttention
import transformer_engine_torch as tex
from transformer_engine.pytorch.fp8 import fp8_autocast
from transformer_engine.common.recipe import DelayedScaling
import torch.utils.benchmark as benchmark

# Configuration class for benchmark parameters
class BenchConfig:
    def __init__(
        self,
        batch_size=32,
        max_seqlen_q=1024,
        max_seqlen_kv=1024,
        num_heads=16,
        head_dim_qk=64,
        num_gqa_groups=16,
        dropout_p=0.0,
        attn_mask_type="causal",
        window_size=None,
        attn_bias_type="no_bias",
    ):
        self.batch_size = batch_size
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.num_gqa_groups = num_gqa_groups
        self.dropout_p = dropout_p
        self.attn_mask_type = attn_mask_type
        self.window_size = window_size
        self.attn_bias_type = attn_bias_type


def benchmark_dpa(
    dtype="bf16",
    qkv_format="bshd",
    kernel_backend="FlashAttention",
    batch_size=32,
    seq_len=1024,
    num_heads=16,
    head_dim=64,
    gqa_groups=None,
    attn_mask_type="causal",
    window_size=None,
    warmup=5,
    iterations=10,
    use_cp=False,
    cp_comm_type="p2p",
    fp8_mha=False,
):
    """Benchmark DotProductAttention module"""
    
    # Set environment variables for kernel backends
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if kernel_backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    elif kernel_backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
    
    # Setup distributed environment if needed
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        if world_size > 1:
            # device_count = torch.cuda.device_count()
            # device = rank % device_count
            device = os.environ.get("LOCAL_RANK")
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    
    # Set up GQA groups
    if gqa_groups is None:
        gqa_groups = num_heads
    
    # Create configuration
    config = BenchConfig(
        batch_size=batch_size,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
        num_heads=num_heads,
        head_dim_qk=head_dim,
        num_gqa_groups=gqa_groups,
        attn_mask_type=attn_mask_type,
        window_size=window_size,
    )
    
    # Setup CP comm group if needed
    cp_comm_group = None
    if use_cp and world_size > 1:
        cp_comm_ranks = range(world_size)
        cp_comm_group = dist.new_group(cp_comm_ranks, backend="nccl")
        if cp_comm_type == "a2a+p2p":
            assert world_size % 2 == 0, "A2A+P2P requires even world size"
            cp_comm_sub_ranks = [range(i * 2, (i + 1) * 2) for i in range(world_size // 2)]
            cp_comm_sub_ranks += [range(i, world_size, 2) for i in range(2)]
            cp_comm_sub_groups = []
            for sub_ranks in cp_comm_sub_ranks:
                sub_group = dist.new_group(sub_ranks, backend="nccl")
                if rank in sub_ranks:
                    cp_comm_sub_groups.append(sub_group)
    
    # Set up FP8 if needed
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}
    fp8_recipe = None
    if dtype == "fp8":
        fp8_recipe = DelayedScaling(fp8_dpa=True, fp8_mha=fp8_mha)
    
    # Create DotProductAttention module
    core_attn = DotProductAttention(
        config.num_heads,
        config.head_dim_qk,
        num_gqa_groups=config.num_gqa_groups,
        attention_dropout=config.dropout_p,
        qkv_format=qkv_format,
        attn_mask_type=config.attn_mask_type,
        window_size=config.window_size,
    ).cuda()
    
    # Create input tensors based on qkv_format
    if qkv_format == "bshd":
        q_input_shape = (
            config.batch_size,
            config.max_seqlen_q // world_size,
            config.num_heads,
            config.head_dim_qk,
        )
        kv_input_shape = (
            config.batch_size,
            config.max_seqlen_kv // world_size,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        attn_output_shape = (
            config.batch_size,
            config.max_seqlen_q // world_size,
            config.num_heads * config.head_dim_qk,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        cu_seqlens_q_padded = None
        cu_seqlens_kv_padded = None
    elif qkv_format == "sbhd":
        q_input_shape = (
            config.max_seqlen_q // world_size,
            config.batch_size,
            config.num_heads,
            config.head_dim_qk,
        )
        kv_input_shape = (
            config.max_seqlen_kv // world_size, 
            config.batch_size,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        attn_output_shape = (
            config.max_seqlen_q // world_size,
            config.batch_size,
            config.num_heads * config.head_dim_qk,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        cu_seqlens_q_padded = None
        cu_seqlens_kv_padded = None
    elif qkv_format == "thd":
        q_input_shape = (
            config.batch_size * config.max_seqlen_q // world_size,
            config.num_heads,
            config.head_dim_qk,
        )
        kv_input_shape = (
            config.batch_size * config.max_seqlen_q // world_size,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        attn_output_shape = (
            config.batch_size * config.max_seqlen_q // world_size,
            config.num_heads * config.head_dim_qk,
        )
        seqlens_q = torch.ones([config.batch_size], dtype=torch.int32) * config.max_seqlen_q
        seqlens_q_padded = (seqlens_q + 2 * world_size - 1) // (world_size * 2) * (world_size * 2)
        cu_seqlens_q_padded = torch.cat(
            [
                torch.zeros([1], dtype=torch.int32),
                seqlens_q_padded.cumsum(0, dtype=torch.int32),
                torch.tensor([q_input_shape[0]], dtype=torch.int32),
            ]
        ).cuda()
        cu_seqlens_q = torch.clone(cu_seqlens_q_padded)
        cu_seqlens_q[-1] = cu_seqlens_q[-2]
        cu_seqlens_kv = cu_seqlens_q
        cu_seqlens_kv_padded = cu_seqlens_q_padded
    else:
        raise ValueError(f"{qkv_format} is an unsupported qkv_format!")

    # Create input tensors
    q = torch.randn(q_input_shape, dtype=dtypes[dtype]).cuda()
    k = torch.randn(kv_input_shape, dtype=dtypes[dtype]).cuda()
    v = torch.randn(kv_input_shape, dtype=dtypes[dtype]).cuda()
    
    # Create bias if needed
    if config.attn_bias_type not in ["no_bias", "alibi"]:
        attn_bias_shape = (1, 1, config.max_seqlen_q, config.max_seqlen_kv)
        bias = torch.randn(*attn_bias_shape, dtype=dtypes[dtype]).cuda()
    else:
        bias = None
    
    # Setup CP if needed
    if use_cp and world_size > 1:
        core_attn.set_context_parallel_group(
            cp_comm_sub_groups if cp_comm_type == "a2a+p2p" else cp_comm_group,
            range(world_size),
            torch.cuda.Stream(),
            cp_comm_type
        )
    
    # Prepare for benchmarking
    torch.cuda.synchronize()
    def run_bench_fwd(q, k, v):
        with (
            fp8_autocast(enabled=dtype == "fp8", fp8_recipe=fp8_recipe)
            if dtype == "fp8"
            else nullcontext()
        ):
            out = core_attn(
                q,
                k,
                v,
                core_attention_bias_type=config.attn_bias_type,
                core_attention_bias=bias,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            )
        return out

    def run_bench_fwd_bwd(q, k, v, dout):
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        
        with (
            fp8_autocast(enabled=dtype == "fp8", fp8_recipe=fp8_recipe)
            if dtype == "fp8"
            else nullcontext()
        ):
            out = core_attn(
                q,
                k,
                v,
                core_attention_bias_type=config.attn_bias_type,
                core_attention_bias=bias,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            )
        out.backward(dout)
    
    dout = torch.randn_like(q.flatten(-2, -1))

# Benchmark forward pass
    f_bench = benchmark.Timer(
        stmt='run_bench_fwd(q, k, v)',
        globals={'run_bench_fwd': run_bench_fwd, 'q': q, 'k': k, 'v': v},
        num_threads=torch.get_num_threads(),
    )
    fwd_time = f_bench.timeit(iterations).mean

# Benchmark forward+backward pass
    fwd_bwd_bench = benchmark.Timer(
        stmt='run_bench_fwd_bwd(q, k, v, dout)',
        globals={'run_bench_fwd_bwd': run_bench_fwd_bwd, 'q': q, 'k': k, 'v': v, 'dout': dout},
        num_threads=torch.get_num_threads(),
    )
    fwd_bwd_time = fwd_bwd_bench.timeit(iterations).mean
    
    # dout = torch.randn(*attn_output_shape, dtype=dtypes[dtype]).cuda()
    # Calculate memory
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with (
        fp8_autocast(enabled=dtype == "fp8", fp8_recipe=fp8_recipe)
        if dtype == "fp8"
        else nullcontext()
    ):
        out = core_attn(
            q,
            k,
            v,
            core_attention_bias_type=config.attn_bias_type,
            core_attention_bias=bias,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
        )
        out.backward(dout)
    torch.cuda.synchronize()
    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Calculate FLOPs for attention
    # FLOPS = 2 * B * H * S_q * S_kv * D
    seq_q = config.max_seqlen_q
    seq_kv = config.max_seqlen_kv

    flops_per_token = 4 * seq_kv * head_dim / (2 if config.attn_mask_type == "causal" else 1)
    total_tokens = batch_size * seq_q
    total_flops = flops_per_token * total_tokens * num_heads * 3.5 # F(1)+ B(2.5)
    flops_per_sec = total_flops / world_size / (fwd_bwd_time)/ (10**12)  # TFLOPs
    
    # Print results
    if rank == 0:
        print(f"\n{'=' * 50}")
        print(f"Benchmark Results for DotProductAttention")
        print(f"{'=' * 50}")
        print(f"Configuration:")
        print(f"  - Dtype: {dtype}")
        print(f"  - QKV Format: {qkv_format}")
        print(f"  - Kernel Backend: {kernel_backend}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Sequence Length: {seq_len}")
        print(f"  - Num Heads: {num_heads}")
        print(f"  - Head Dimension: {head_dim}")
        print(f"  - GQA Groups: {gqa_groups}")
        print(f"  - Attention Mask: {attn_mask_type}")
        print(f"  - Context Parallel: {use_cp} (Type: {cp_comm_type if use_cp else 'N/A'})")
        print(f"  - FP8 MHA: {fp8_mha}")
        print(f"  - World Size: {world_size}")
        print(f"\nPerformance:")
        print(f"  - Forward Time: {fwd_time*1000:.2f} ms")
        print(f"  - Forward+Backward Time: {fwd_bwd_time*1000:.2f} ms")
        print(f"  - Memory Usage: {memory_usage:.2f} MB")
        print(f"  - Performance: {flops_per_sec:.2f} TFLOPs/s")
        print(f"{'=' * 50}\n")
    
    # Return metrics as a dictionary
    return {
        "forward_time_ms": fwd_time * 1000,
        "fwd_bwd_time_ms": fwd_bwd_time * 1000,
        "memory_mb": memory_usage,
        "tflops": flops_per_sec,
    }


def run_benchmarks(args):
    """Run benchmarks with different configurations"""
    results = []
    
    # Initialize distributed environment if needed
    # if args.world_size >= 1 and not dist.is_initialized():
        # Initialize distributed using environment variables
        # os.environ["RANK"] = "0"
        # os.environ["WORLD_SIZE"] = str(args.world_size)
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    args.world_size = dist.get_world_size()
    
    # Run benchmarks
    for dtype in args.dtypes:
        for qkv_format in args.qkv_formats:
            for kernel in args.kernels:
                for seq_len in args.seq_lens:
                    for batch_size in args.batch_sizes:
                        for use_cp in args.context_parallel:
                            # Skip CP if world_size is 1
                            if use_cp and (args.world_size == 1 or dist.get_world_size() == 1):
                                continue
                                
                            metric = benchmark_dpa(
                                dtype=dtype,
                                qkv_format=qkv_format,
                                kernel_backend=kernel,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                num_heads=args.num_heads,
                                head_dim=args.head_dim,
                                gqa_groups=args.gqa_groups,
                                attn_mask_type=args.attn_mask_type,
                                window_size=args.window_size,
                                warmup=args.warmup,
                                iterations=args.iterations,
                                use_cp=use_cp,
                                cp_comm_type=args.cp_comm_type,
                                fp8_mha=args.fp8_mha,
                            )
                            
                            # Save results
                            result = {
                                "dtype": dtype,
                                "qkv_format": qkv_format,
                                "kernel": kernel,
                                "batch_size": batch_size,
                                "seq_len": seq_len,
                                "num_heads": args.num_heads,
                                "head_dim": args.head_dim,
                                "context_parallel": use_cp,
                                "world_size": args.world_size,
                                **metric
                            }
                            results.append(result)
    
    # Print summary table if rank 0
    if dist.get_rank() == 0:
        print("\nBenchmark Summary:")
        print(f"{'=' * 120}")
        header = f"{'Dtype':^8} | {'QKV Format':^8} | {'Kernel':^14} | {'B':^4} | {'Seq Len':^7} | {'CP':^5} | {'FWD (ms)':^10} | {'FWD+BWD (ms)':^12} | {'Memory (MB)':^12} | {'TFLOPs/s':^10}"
        print(header)
        print(f"{'-' * 120}")
        
        for result in results:
            row = (
                f"{result['dtype']:^8} | "
                f"{result['qkv_format']:^8} | "
                f"{result['kernel']:^14} | "
                f"{result['batch_size']:^4} | "
                f"{result['seq_len']:^7} | "
                f"{str(result['context_parallel']):^5} | "
                f"{result['forward_time_ms']:^10.2f} | "
                f"{result['fwd_bwd_time_ms']:^12.2f} | "
                f"{result['memory_mb']:^12.2f} | "
                f"{result['tflops']:^10.2f}"
            )
            print(row)
        
        print(f"{'=' * 120}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DotProductAttention")
    
    # Model parameters
    parser.add_argument("--dtypes", nargs="+", default=["bf16"], 
                        choices=["fp16", "bf16", "fp8"], help="Data types to benchmark")
    parser.add_argument("--qkv-formats", nargs="+", default=["bshd"], 
                        choices=["bshd", "sbhd", "thd"], help="QKV formats to benchmark")
    parser.add_argument("--kernels", nargs="+", default=["FlashAttention"], 
                        choices=["FlashAttention", "FusedAttention"], help="Attention kernels to benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32], 
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1024], 
                        help="Sequence lengths to benchmark")
    parser.add_argument("--num-heads", type=int, default=16, 
                        help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, 
                        help="Dimension of each head")
    parser.add_argument("--gqa-groups", type=int, default=None, 
                        help="Number of GQA groups (defaults to num_heads if None)")
    parser.add_argument("--attn-mask-type", default="causal", 
                        choices=["causal", "no_mask"], help="Attention mask type")
    parser.add_argument("--window-size", type=int, default=None, 
                        help="Window size for sliding window attention")
    
    # Parallelism parameters
    parser.add_argument("--world-size", type=int, default=1, 
                        help="World size for distributed training")
    parser.add_argument("--context-parallel", nargs="+", type=lambda x: x.lower() == "true", 
                        default=[False], help="Whether to use context parallelism")
    parser.add_argument("--cp-comm-type", default="p2p", 
                        choices=["p2p", "a2a+p2p", "a2a"], help="Communication pattern for context parallelism")
    
    # FP8 parameters
    parser.add_argument("--fp8-mha", action="store_true", 
                        help="Whether to use FP8 in MHA")
    
    # Benchmark parameters
    parser.add_argument("--warmup", type=int, default=5, 
                        help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, 
                        help="Number of benchmark iterations")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_benchmarks(args)

