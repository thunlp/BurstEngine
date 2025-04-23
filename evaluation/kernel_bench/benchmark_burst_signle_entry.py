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
import numpy as np
import jsonlines as jl
from contextlib import nullcontext

# Import burst attention functionality
from burst_attn import burst_attn_func, burst_attn_func_striped
from burst_attn.comm import get_world_size, print_rank, get_rank

def single_all_to_all(input, scatter_idx, gather_idx, group):
    seq_world_size = torch.distributed.get_world_size(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + inp_shape[scatter_idx + 1 :]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        _reshape = [-1, seq_world_size, inp_shape[scatter_idx]] + inp_shape[
            scatter_idx + 1 :
        ]

        input_t = input.reshape(_reshape).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    torch.distributed.all_to_all_single(output, input_t, group=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_idx < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[:gather_idx]
        + [
            inp_shape[gather_idx] * seq_world_size,
        ]
        + inp_shape[gather_idx + 1 :]
    ).contiguous()


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, scatter_idx, gather_idx):
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return single_all_to_all(input, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            None,
            _SeqAllToAll.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx
            ),
            None,
            None,
        )


def generate_inp(batch, seq_len, num_heads, head_dim):
    """Generate input tensors for attention."""
    q = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda")
    k = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda")
    v = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda")
    return q, k, v


def get_local_global_groups(local_size):
    """Create local and global communication groups."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Create intra-node groups
    group_ranks = np.array(list(range(world_size)))
    intra_ranks = group_ranks.reshape(-1, local_size)
    inter_ranks = intra_ranks.transpose()
    
    intra_group, _ = dist.new_subgroups_by_enumeration(
        intra_ranks.tolist(), backend="nccl"
    )
    
    inter_group, _ = dist.new_subgroups_by_enumeration(
        inter_ranks.tolist(), backend="nccl"
    )
    
    # Create secondary groups for dkv
    intra_group2, _ = dist.new_subgroups_by_enumeration(
        intra_ranks.tolist(), backend="nccl"
    )
    
    inter_group2, _ = dist.new_subgroups_by_enumeration(
        inter_ranks.tolist(), backend="nccl"
    )
    
    return (
        None,
        (intra_group, intra_group2),
        (inter_group, inter_group2)
    )


def flash_attn(q, k, v, causal=False, **kwargs):
    """Flash attention implementation"""
    try:
        from flash_attn import flash_attn_func
        output = flash_attn_func(q, k, v, causal=causal)
        return output
    except ImportError:
        print("Flash attention not available, skipping.")
        return torch.zeros_like(q)


def ref_attn(q, k, v, **kwargs):
    """Reference attention implementation"""
    # Standard attention: Q * K^T * V
    s = q.shape[0]
    q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output


def ring_attn(q, k, v, **kwargs):
    """Ring attention implementation"""
    # Implementation would depend on specific ring attention logic
    # Placeholder for actual implementation
    return ref_attn(q, k, v)


def burst_attn_impl(q, k, v, group_info, causal=False, opt_bwd=True, double_ring=False, is_striped=False):
    """Burst attention implementation wrapper"""
    group, intra_groups, inter_groups = group_info if double_ring else (None, None, None)
    
    func = burst_attn_func if not is_striped else burst_attn_func_striped
    
    return func(
        q, k, v, None, "cuda", causal, opt_bwd, False,
        group, [intra_groups, inter_groups]
    )


class BenchConfig:
    def __init__(
        self,
        batch_size=32,
        seq_len=1024,
        num_heads=16,
        head_dim=64,
        causal=True,
        double_ring=False,
        opt_bwd=True,
        use_striped=False
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal
        self.double_ring = double_ring
        self.opt_bwd = opt_bwd
        self.use_striped = use_striped


def calculate_flops(batch, seq_len, num_heads, head_dim, causal):
    """Calculate FLOPs for attention operation."""
    # FLOPs for attention: 4 * b * s^2 * h * d / (2 if causal else 1)
    flops_per_token = 4 * seq_len * head_dim / (2 if causal else 1)
    total_tokens = batch * seq_len
    # Forward (1x) + backward (2.5x) for total of 3.5x
    total_flops = flops_per_token * total_tokens * num_heads * 3.5
    return total_flops


def benchmark_burst(
    dtype="bf16",
    batch_size=32,
    seq_len=1024,
    num_heads=16,
    head_dim=64,
    causal=True,
    double_ring=False,
    opt_bwd=True,
    use_striped=False,
    warmup=5,
    iterations=10,
    profile=False,
):
    """Benchmark burst attention."""
    
    # Setup distributed environment if needed
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        if world_size > 1:
            device = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    
    # Setup local/global groups for double ring
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    group_info = get_local_global_groups(local_size) if double_ring else (None, None, None)
    
    # Set up data types
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    
    # Create configuration
    config = BenchConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        double_ring=double_ring,
        opt_bwd=opt_bwd,
        use_striped=use_striped
    )
    
    # Create input tensors - adjust seq_len for distributed setting
    seq_per_gpu = seq_len // world_size
    q = torch.randn(batch_size, seq_per_gpu, num_heads, head_dim, dtype=dtypes[dtype]).cuda()
    k = torch.randn(batch_size, seq_per_gpu, num_heads, head_dim, dtype=dtypes[dtype]).cuda()
    v = torch.randn(batch_size, seq_per_gpu, num_heads, head_dim, dtype=dtypes[dtype]).cuda()
    
    # Define the attention function to benchmark
    def attn_func(q, k, v):
        return burst_attn_impl(
            q, k, v, 
            group_info=group_info,
            causal=causal,
            opt_bwd=opt_bwd,
            double_ring=double_ring,
            is_striped=use_striped
        )
    
    # Warmup
    # Warmup
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(warmup):
        out = attn_func(q, k, v)
    
    # Benchmark forward pass
    start_event.record()
    for i in range(iterations):
        out = attn_func(q, k, v)
    end_event.record()
    torch.cuda.synchronize()
    fwd_time = start_event.elapsed_time(end_event) / iterations / 1000
    
    # Set requires_grad for backward pass
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    dout = torch.randn_like(q)
    
    # Benchmark forward+backward pass
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iterations):
        out = attn_func(q, k, v)
        out.backward(dout)
    end_event.record()
    torch.cuda.synchronize()
    fwd_bwd_time = start_event.elapsed_time(end_event) / iterations / 1000
    
    if profile:
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1) 
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) 
        with profiler:
            out = attn_func(q, k, v)
            out.backward(dout)
            profiler.step()
        from profile_utils import get_summary
        summary, kernel_times = get_summary(profiler)
    else:
        summary = None

    if summary is None:
        flash_t = torch.nan
        sendrecv_t = torch.nan
    else:
        flash_t = summary["flash"]
        sendrecv_t = summary["SendRecv"]
    if torch.distributed.get_rank() == 0:
        print(kernel_times)
    # Calculate memory usage
    torch.cuda.reset_peak_memory_stats()
    out = attn_func(q, k, v)
    out.backward(dout)
    torch.cuda.synchronize()
    memory_usage = torch.cuda.max_memory_allocated() / (1024**2)  # MB
 
    # Calculate FLOPs
    total_flops = calculate_flops(batch_size, seq_len, num_heads, head_dim, causal)
    flops_per_sec = total_flops / world_size / fwd_bwd_time / (10**12)  # TFLOPs
    
    # Print results
    if rank == 0:
        print(f"\n{'=' * 50}")
        print(f"Benchmark Results for Burst Attention")
        print(f"{'=' * 50}")
        print(f"Configuration:")
        print(f"  - Dtype: {dtype}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Sequence Length: {seq_len}")
        print(f"  - Num Heads: {num_heads}")
        print(f"  - Head Dimension: {head_dim}")
        print(f"  - Causal: {causal}")
        print(f"  - Double Ring: {double_ring}")
        print(f"  - Optimize Backward: {opt_bwd}")
        print(f"  - Striped: {use_striped}")
        print(f"  - World Size: {world_size}")
        print(f"\nPerformance:")
        print(f"  - Forward Time: {fwd_time * 1000:.2f} ms")
        print(f"  - Forward+Backward Time: {fwd_bwd_time * 1000:.2f} ms")
        print(f"  - FlashAttention Kernel: {flash_t:.2f} %")
        print(f"  - RingComm : {sendrecv_t:.2f} %")
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
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    
    # Open results file
    result_file = None
    if dist.get_rank() == 0 and args.save_results:
        result_file = jl.open(args.output_file, "a")
    
    # Run benchmarks
    for dtype in args.dtypes:
        for seq_len in args.seq_lens:
            for batch_size in args.batch_sizes:
                for double_ring in args.double_ring:
                    # Skip double_ring if world_size is 1
                    if double_ring and dist.get_world_size() == 1:
                        continue
                    
                    for opt_bwd in args.opt_bwd:
                        for use_striped in args.use_striped:
                            # Skip non-causal with striped implementation
                            if use_striped and not args.causal:
                                continue
                                
                            metric = benchmark_burst(
                                dtype=dtype,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                num_heads=args.num_heads,
                                head_dim=args.head_dim,
                                causal=args.causal,
                                double_ring=double_ring,
                                opt_bwd=opt_bwd,
                                use_striped=use_striped,
                                warmup=args.warmup,
                                profile=args.profile,
                                iterations=args.iterations
                            )
                            
                            # Save results
                            result = {
                                "dtype": dtype,
                                "batch_size": batch_size,
                                "seq_len": seq_len,
                                "num_heads": args.num_heads,
                                "head_dim": args.head_dim,
                                "causal": args.causal,
                                "double_ring": double_ring,
                                "opt_bwd": opt_bwd,
                                "use_striped": use_striped,
                                "world_size": dist.get_world_size(),
                                **metric,
                            }
                            results.append(result)
                            
                            # Write to file if enabled
                            if dist.get_rank() == 0 and args.save_results and result_file:
                                result_file.write(result)
    
    # Close results file
    if dist.get_rank() == 0 and args.save_results and result_file:
        result_file.close()
    
    # Print summary table if rank 0
    if dist.get_rank() == 0:
        print("\nBenchmark Summary:")
        print(f"{'=' * 130}")
        header = (
            f"{'Dtype':^8} | {'B':^4} | {'Seq Len':^7} | {'Heads':^5} | "
            f"{'DR':^5} | {'Opt BWD':^7} | {'Striped':^7} | "
            f"{'FWD (ms)':^10} | {'FWD+BWD (ms)':^12} | {'Memory (MB)':^12} | {'TFLOPs/s':^10}"
        )
        print(header)
        print(f"{'-' * 130}")
        
        for result in results:
            # DR = Double Ring
            row = (
                f"{result['dtype']:^8} | "
                f"{result['batch_size']:^4} | "
                f"{result['seq_len']:^7} | "
                f"{result['num_heads']:^5} | "
                f"{str(result['double_ring'])[0]:^5} | "
                f"{str(result['opt_bwd'])[0]:^7} | "
                f"{str(result['use_striped'])[0]:^7} | "
                f"{result['forward_time_ms']:^10.2f} | "
                f"{result['fwd_bwd_time_ms']:^12.2f} | "
                f"{result['memory_mb']:^12.2f} | "
                f"{result['tflops']:^10.2f}"
            )
            print(row)
        
        print(f"{'=' * 130}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Burst Attention")
    
    # Model parameters
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["bf16"],
        choices=["fp16", "bf16", "fp32"],
        help="Data types to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[32],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[1024],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--num-heads", 
        type=int, 
        default=16, 
        help="Number of attention heads"
    )
    parser.add_argument(
        "--head-dim", 
        type=int, 
        default=64, 
        help="Dimension of each head"
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        default=True,
        help="Use causal attention",
    )
    
    # Parallelism parameters
    parser.add_argument(
        "--double-ring",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[False, True],
        help="Whether to use double ring pattern",
    )
    parser.add_argument(
        "--opt-bwd",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[True],
        help="Whether to optimize backward pass",
    )
    parser.add_argument(
        "--use-striped",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[False, True],
        help="Whether to use striped version (causal only)",
    )
    
    # Benchmark parameters
    parser.add_argument(
        "--warmup", 
        type=int, 
        default=5, 
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=10, 
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to a file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results_burst.jsonl",
        help="Output file for results",
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the benchmark",

    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_benchmarks(args)

