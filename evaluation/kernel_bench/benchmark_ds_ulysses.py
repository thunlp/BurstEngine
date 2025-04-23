#!/usr/bin/env python3

import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
from contextlib import nullcontext
import numpy as np
from torch import Tensor
from typing import Any,  Tuple
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except ImportError:
    flash_attn_qkvpacked_func = None
    flash_attn_func = None

def single_all_to_all(input, scatter_idx, gather_idx, group):
    seq_world_size = torch.distributed.get_world_size(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        _reshape = [-1, seq_world_size, inp_shape[scatter_idx]] + inp_shape[scatter_idx + 1:]

        input_t = input.reshape(_reshape).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    torch.distributed.all_to_all_single(output, input_t, group=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_idx < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[: gather_idx] + \
        [inp_shape[gather_idx] * seq_world_size,] + \
        inp_shape[gather_idx + 1:]).contiguous()


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return single_all_to_all(input, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx, *grad_output: Tensor):
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)
# Simplified single_all_to_all function for the benchmark


# Configuration class for benchmark parameters
class BenchConfig:
    def __init__(
        self,
        batch_size=32,
        max_seqlen_q=1024,
        max_seqlen_kv=1024,
        num_heads=16,
        head_dim_qk=64,
        dropout_p=0.0,
        attn_mask_type="causal",
    ):
        self.batch_size = batch_size
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.dropout_p = dropout_p
        self.attn_mask_type = attn_mask_type

# Ulysses Attention implementation
class UlyssesAttention(torch.nn.Module):
    def __init__(self, num_heads, head_dim, seq_parallel_group):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        self.seq_parallel_group = seq_parallel_group
        
    def forward(self, q, k, v, attention_mask=None):
        batch_size, seq_length, *_ = q.shape
        
        # Scatter q across sequence parallel devices, gather heads
        scatter_idx = 2  # Sequence dimension
        gather_idx = 1  # Head dimension
        batch_dim_idx = 0  # Batch dimension
        
        # All2All: sequence parallel to head parallel
        q_s2h = _SeqAllToAll.apply(self.seq_parallel_group, q, scatter_idx, gather_idx) 
        k_s2h = _SeqAllToAll.apply(self.seq_parallel_group, k, scatter_idx, gather_idx) 
        v_s2h = _SeqAllToAll.apply(self.seq_parallel_group, v, scatter_idx, gather_idx) 
        
        # Compute attention scores with FlashAttention
        # q_s2h = q_s2h.transpose(1, 2).contiguous()  # [batch, seq, head, dim]
        # k_s2h = k_s2h.transpose(1, 2).contiguous()  # [batch, seq, head, dim]
        # v_s2h = v_s2h.transpose(1, 2).contiguous()  # [batch, seq, head, dim]
        
        # Apply FlashAttention

        attn_output = flash_attn_func(
            q_s2h, k_s2h, v_s2h, 
            dropout_p=0.0, 
            causal=True
        )
        
        # Reshape back
        attn_output = attn_output.contiguous()  # [batch, head, seq, dim]
        
        # All2All: head parallel back to sequence parallel
        attn_output = _SeqAllToAll.apply(self.seq_parallel_group, attn_output, gather_idx, scatter_idx)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        return attn_output

def benchmark_ulysses(
    dtype="bf16",
    batch_size=32,
    seq_len=1024,
    num_heads=16,
    head_dim=64,
    attn_mask_type="causal",
    seq_parallel_size=2,
    warmup=5,
    iterations=10,
):
    """Benchmark Ulysses attention mechanism"""
    
    # Setup distributed environment
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    
    # Check if world_size is compatible with seq_parallel_size
    if world_size < seq_parallel_size:
        print(f"Warning: world_size ({world_size}) < seq_parallel_size ({seq_parallel_size})")
        seq_parallel_size = world_size
    
    # Create sequence parallel group
    seq_parallel_group = dist.new_group(list(range(seq_parallel_size)), backend="nccl")
    
    # Set device
    device = torch.cuda.current_device()
    
    # Create configuration
    config = BenchConfig(
        batch_size=batch_size,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
        num_heads=num_heads,
        head_dim_qk=head_dim,
        attn_mask_type=attn_mask_type,
    )
    
    # Check if FlashAttention is available
    if flash_attn_func is None:
        if rank == 0:
            print("Warning: FlashAttention is not available. Falling back to standard attention.")
    
    # Create Ulysses attention module
    ulysses_attn = UlyssesAttention(config.num_heads, config.head_dim_qk, seq_parallel_group).to(device)
    
    # Convert to appropriate dtype
    if dtype == "fp16":
        ulysses_attn = ulysses_attn.half()
    elif dtype == "bf16":
        ulysses_attn = ulysses_attn.bfloat16()
    
    # Create input tensors
    input_shape = (config.batch_size, config.max_seqlen_q // seq_parallel_size, config.num_heads , config.head_dim_qk)
    qkv = [torch.randn(input_shape, device=device) for _ in range(3)]
    
    # Convert to appropriate dtype
    for i in range(3):
        if dtype == "fp16":
            qkv[i] = qkv[i].half()
        elif dtype == "bf16":
            qkv[i] = qkv[i].bfloat16()
    
    # Prepare for benchmarking
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            output = ulysses_attn(*qkv)
    
    # Benchmark forward pass
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            output = ulysses_attn(*qkv)
    torch.cuda.synchronize()
    fwd_time = (time.time() - start) / iterations
    
    # Set requires_grad for backward pass
    
    for i in qkv:
        i.requires_grad_(True)
    for param in ulysses_attn.parameters():
        param.requires_grad_(True)
    
    # Create gradient tensor
    grad_output = torch.randn_like(output)
    
    # Benchmark forward+backward pass
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        output = ulysses_attn(*qkv)
        output.backward(grad_output)
    torch.cuda.synchronize()
    fwd_bwd_time = (time.time() - start) / iterations
    
    # Calculate memory usage
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    output = ulysses_attn(*qkv)
    output.backward(grad_output)
    torch.cuda.synchronize()
    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Calculate FLOPs for attention
    # FLOPS = 2 * B * H * S_q * S_kv * D
    flops_per_token = 4 * config.max_seqlen_q * config.head_dim_qk / (2 if config.attn_mask_type == "causal" else 1)
    total_tokens = config.batch_size * config.max_seqlen_q
    total_flops = flops_per_token * total_tokens * config.num_heads * 3.5  # F(1) + B(2.5)
    flops_per_sec = total_flops / seq_parallel_size / (fwd_bwd_time) / (10**12)  # TFLOPs
    
    # Print results
    if rank == 0:
        print(f"\n{'=' * 50}")
        print(f"Benchmark Results for Ulysses Attention")
        print(f"{'=' * 50}")
        print(f"Configuration:")
        print(f"  - Dtype: {dtype}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Sequence Length: {seq_len}")
        print(f"  - Num Heads: {num_heads}")
        print(f"  - Head Dimension: {head_dim}")
        print(f"  - Attention Mask: {attn_mask_type}")
        print(f"  - Sequence Parallel Size: {seq_parallel_size}")
        print(f"  - FlashAttention: {'Available' if flash_attn_func is not None else 'Not Available'}")
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
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12345")
    
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    
    # Run benchmarks
    for dtype in args.dtypes:
        for seq_len in args.seq_lens:
            for batch_size in args.batch_sizes:
                    seq_parallel_size = world_size
                    metric = benchmark_ulysses(
                        dtype=dtype,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_heads=args.num_heads,
                        head_dim=args.head_dim,
                        attn_mask_type=args.attn_mask_type,
                        seq_parallel_size=seq_parallel_size,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                    
                    # Save results
                    result = {
                        "dtype": dtype,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "num_heads": args.num_heads,
                        "head_dim": args.head_dim,
                        "seq_parallel_size": seq_parallel_size,
                        "world_size": world_size,
                        **metric
                    }
                    results.append(result)
    
    # Print summary table if rank 0
    if rank == 0:
        print("\nBenchmark Summary:")
        print(f"{'=' * 120}")
        header = f"{'Dtype':^8} | {'B':^4} | {'Seq Len':^7} | {'SP Size':^7} | {'FWD (ms)':^10} | {'FWD+BWD (ms)':^12} | {'Memory (MB)':^12} | {'TFLOPs/s':^10}"
        print(header)
        print(f"{'-' * 120}")
        
        for result in results:
            row = (
                f"{result['dtype']:^8} | "
                f"{result['batch_size']:^4} | "
                f"{result['seq_len']:^7} | "
                f"{result['seq_parallel_size']:^7} | "
                f"{result['forward_time_ms']:^10.2f} | "
                f"{result['fwd_bwd_time_ms']:^12.2f} | "
                f"{result['memory_mb']:^12.2f} | "
                f"{result['tflops']:^10.2f}"
            )
            print(row)
        
        print(f"{'=' * 120}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Ulysses Attention")
    
    # Model parameters
    parser.add_argument("--dtypes", nargs="+", default=["bf16"], 
                        choices=["fp16", "bf16"], help="Data types to benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32], 
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1024], 
                        help="Sequence lengths to benchmark")
    parser.add_argument("--num-heads", type=int, default=16, 
                        help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, 
                        help="Dimension of each head")
    parser.add_argument("--attn-mask-type", default="causal", 
                        choices=["causal", "no_mask"], help="Attention mask type")
    
    # Parallelism parameters
    parser.add_argument("--world-size", type=int, default=1, 
                        help="World size for distributed training")
    
    # Benchmark parameters
    parser.add_argument("--warmup", type=int, default=5, 
                        help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, 
                        help="Number of benchmark iterations")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    results = run_benchmarks(args)
