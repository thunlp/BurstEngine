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
import torch.utils.benchmark as benchmark

# Import the ZigZagRingFlashAttnFunc from benchmark_doublering.py
import sys

sys.path.append("../baselines/InternEvo/")
from internlm.core.context import Config
from internlm.core.context import global_context as gpc
from internlm.model.ops.attention import (
    zigzag_ring_flash_attn_qkvsplited_func_with_sliding_window,
)
from torch import Tensor
def print_group_info(groups):
    if torch.distributed.get_rank() == 0:
        for g in groups:
            print(torch.distributed.get_process_group_ranks(g), end="\t")

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
    def forward(ctx, group, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        torch.cuda.nvtx.range_push("_SeqAllToAll")
        res = single_all_to_all(input, scatter_idx, gather_idx, group)
        torch.cuda.nvtx.range_pop()
        return res

    @staticmethod
    def backward(ctx, *grad_output: Tensor):
        return (
            None,
            _SeqAllToAll.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx
            ),
            None,
            None,
        )


def attn_impl(
    q,
    k,
    v,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    alibi_slopes,
    deterministic,
    return_attn_probs,
    context_group,
    inter_window_group,
    intra_window_group,
    dkv_inter_window_group,
    dkv_intra_window_group,
    layer_idx,
    use_ulysses=False,
    head_first=False,
):
    if not use_ulysses:
        # if torch.distributed.get_rank() == 0:
        #     print("inter intra")
        # print_group_info([inter_window_group,intra_window_group])
        out = zigzag_ring_flash_attn_qkvsplited_func_with_sliding_window(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=None,
            causal=causal,
            window_size=window_size,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            context_group=context_group,
            inter_window_group=inter_window_group,
            intra_window_group=intra_window_group,
            dkv_inter_window_group=dkv_inter_window_group,
            dkv_intra_window_group=dkv_intra_window_group,
            layer_idx=0,
        )
    else:
        batch, seq_len, num_heads, head_dim = q.shape
        scatter_idx = 2  # Sequence dimension
        gather_idx = 1  # Head dimension
        if os.environ["LOCAL_WORLD_SIZE"] == os.environ['WORLD_SIZE']:
            head_first = True
        if head_first:
            hp_group = intra_window_group
            cp_group = inter_window_group
            dkv_cp_group = dkv_inter_window_group
        else:
            hp_group = inter_window_group
            cp_group = intra_window_group
            dkv_cp_group = dkv_intra_window_group

        q_s2h = _SeqAllToAll.apply(hp_group, q, scatter_idx, gather_idx)
        k_s2h = _SeqAllToAll.apply(hp_group, k, scatter_idx, gather_idx)
        v_s2h = _SeqAllToAll.apply(hp_group, v, scatter_idx, gather_idx)
        attn_output = zigzag_ring_flash_attn_qkvsplited_func_with_sliding_window(
            q_s2h,
            k_s2h,
            v_s2h,
            dropout_p=dropout_p,
            softmax_scale=None,
            causal=causal,
            window_size=window_size,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            context_group=cp_group,
            inter_window_group=cp_group,
            intra_window_group=cp_group,
            dkv_inter_window_group=dkv_cp_group,
            dkv_intra_window_group=dkv_cp_group,
            layer_idx=0,
        )
        out = _SeqAllToAll.apply(
            hp_group,
            attn_output,
            gather_idx,
            scatter_idx,
        )

    return out


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
        use_ulysses=False,
        head_first=False
    ):
        self.batch_size = batch_size
        self.use_ulysses = use_ulysses
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.num_gqa_groups = num_gqa_groups
        self.dropout_p = dropout_p
        self.attn_mask_type = attn_mask_type
        self.window_size = window_size
        self.head_first = head_first
        self.use_ulysses = use_ulysses
        self.attn_bias_type = attn_bias_type


def benchmark_zigzag(
    dtype="bf16",
    qkv_format="bshd",
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
    use_ulysses=False,
    head_first=False,
    profile=False,
):
    """Benchmark ZigZagRingFlashAttnFunc"""

    # Setup distributed environment if needed
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        if world_size > 1:
            device = os.environ.get("LOCAL_RANK")
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # Set up GQA groups
    if gqa_groups is None:
        gqa_groups = num_heads
    gpc.is_forward = True

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
        use_ulysses=use_ulysses,
        head_first=head_first,
    )

    # Setup CP comm group if needed
    context_group = None

    inter_window_group = None
    intra_window_group = None
    dkv_inter_window_group = None
    dkv_intra_window_group = None

    if use_cp and world_size > 1:
        # Create communication groups
        cp_comm_ranks = range(world_size)
        context_group = dist.new_group(cp_comm_ranks, backend="nccl")

        # Determine number of nodes and local size
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        nnodes = world_size // local_size

        # Create proper inter/intra node communication groups
        group_ranks = np.array(list(range(world_size)))
        intra_ranks = group_ranks.reshape(-1, local_size)
        inter_ranks = intra_ranks.transpose()

        # Create primary groups
        intra_window_group, _ = dist.new_subgroups_by_enumeration(
            intra_ranks.tolist(), backend="nccl"
        )
        inter_window_group, _ = dist.new_subgroups_by_enumeration(
            inter_ranks.tolist(), backend="nccl"
        )

        # Create secondary groups for dkv
        dkv_intra_window_group, _ = dist.new_subgroups_by_enumeration(
            intra_ranks.tolist(), backend="nccl"
        )
        dkv_inter_window_group, _ = dist.new_subgroups_by_enumeration(
            inter_ranks.tolist(), backend="nccl"
        )

        parallel = dict(
            zero1=dict(size=-1),
            tensor=dict(size=world_size, mode="isp"),
            pipeline=dict(size=1, interleaved_overlap=True),
            weight=dict(size=world_size, overlap=True, memory_pool=False),
            sequence_2D=dict(
                enable=True,
                head_size=1,
                window_size=local_size,
                context_size=world_size,
                device_placement_strategy=dict(head_first=head_first, interleaved=False),
            ),
        )
    gpc._config = Config(dict(selective_checkpoint=False, parallel=parallel))
    # Set up data types
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

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
    else:
        raise ValueError(f"{qkv_format} is an unsupported qkv_format!")

    # Create input tensors
    q = torch.randn(q_input_shape, dtype=dtypes[dtype]).cuda()
    k = torch.randn(kv_input_shape, dtype=dtypes[dtype]).cuda()
    v = torch.randn(kv_input_shape, dtype=dtypes[dtype]).cuda()

    # Set window size parameter
    if window_size is None:
        window_size = (-1, -1)
    else:
        window_size = (window_size, window_size)

    # Prepare for benchmarking
    torch.cuda.synchronize()

    def attn_fn(q, k, v):
        return attn_impl(
                q,
                k,
                v,
                dropout_p=config.dropout_p,
                softmax_scale=None,
                causal=(config.attn_mask_type == "causal"),
                window_size=window_size,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False,
                context_group=context_group,
                inter_window_group=inter_window_group,
                intra_window_group=intra_window_group,
                dkv_inter_window_group=dkv_inter_window_group,
                dkv_intra_window_group=dkv_intra_window_group,
                layer_idx=0,
                use_ulysses=config.use_ulysses,
                head_first=config.head_first,
        )

    def run_benchmark_bwd(q, k, v, grad):
        q.grad = None
        k.grad = None
        v.grad = None
        y= attn_fn(q, k, v)
        if grad is None:
            grad = torch.randn_like(y)
        else:
            assert grad.shape == y.shape
        y.backward(grad)

    f_bench = benchmark.Timer(
        stmt='attn_fn(*qkv)',
        globals={'attn_fn': attn_fn, 'qkv': (q, k, v)},
        num_threads=torch.get_num_threads(),
    )
    fwd_time = f_bench.timeit(iterations).mean


    # Set requires_grad for backward pass
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    # dout = torch.empty_like(out)
    dout = torch.randn_like(q)
    fwd_bwd_bench = benchmark.Timer(
        stmt='out = run_benchmark_bwd(*qkv, dout)',
        globals={'run_benchmark_bwd': run_benchmark_bwd, 'dout': dout, 'qkv': (q, k, v)},
        num_threads=torch.get_num_threads(),
    )
    
    fwd_bwd_time = fwd_bwd_bench.timeit(iterations).mean
    summary = None
    if profile:
        start = time.time()
        schedule = torch.profiler.schedule(wait=0, warmup=3, active=10, repeat=1) 
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], schedule=schedule,profile_memory=True
) 
        with profiler:
            for i in range(13):
                out = attn_impl(
                    q,
                    k,
                    v,
                    dropout_p=config.dropout_p,
                    softmax_scale=None,
                    causal=(config.attn_mask_type == "causal"),
                    window_size=window_size,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=False,
                    context_group=context_group,
                    inter_window_group=inter_window_group,
                    intra_window_group=intra_window_group,
                    dkv_inter_window_group=dkv_inter_window_group,
                    dkv_intra_window_group=dkv_intra_window_group,
                    layer_idx=0,
                    use_ulysses=config.use_ulysses,
                    head_first=config.head_first,
                )
                out.backward(dout)
                profiler.step()
        from profile_utils import get_summary
        summary, kernel_times = get_summary(profiler)
        if torch.distributed.get_rank() == 0:
            print(kernel_times)

    torch.cuda.synchronize()

    # Calculate memory
    torch.cuda.reset_peak_memory_stats()
    out = attn_impl(
        q,
        k,
        v,
        dropout_p=config.dropout_p,
        softmax_scale=None,
        causal=(config.attn_mask_type == "causal"),
        window_size=window_size,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        context_group=context_group,
        inter_window_group=inter_window_group,
        intra_window_group=intra_window_group,
        dkv_inter_window_group=dkv_inter_window_group,
        dkv_intra_window_group=dkv_intra_window_group,
        layer_idx=0,
        use_ulysses=config.use_ulysses,
        head_first=config.head_first,
    )
    out.backward(dout)
    torch.cuda.synchronize()
    memory_usage = torch.cuda.max_memory_allocated() / (1024**2)  # MB

    # if torch.distributed.get_rank() == 0:
    #     print("forward done4")
    # Calculate FLOPs for attention
    seq_q = config.max_seqlen_q
    seq_kv = config.max_seqlen_kv

    flops_per_token = (
        4 * seq_kv * head_dim / (2 if config.attn_mask_type == "causal" else 1)
    )
    total_tokens = batch_size * seq_q
    total_flops = flops_per_token * total_tokens * num_heads * 3.5  # F(1) + B(2.5)
    flops_per_sec = total_flops / world_size / (fwd_bwd_time) / (10**12)  # TFLOPs

    # Print results
    if summary is None:
        flash_t = torch.nan
        sendrecv_t = torch.nan
    else:
        flash_t = summary["flash"]
        sendrecv_t = summary["SendRecv"]
    if rank == 0:
        print(f"\n{'=' * 50}")
        print(f"Benchmark Results for ZigZagRingFlashAttnFunc")
        print(f"{'=' * 50}")
        print(f"Configuration:")
        print(f"  - Dtype: {dtype}")
        print(f"  - QKV Format: {qkv_format}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Sequence Length: {seq_len}")
        print(f"  - Num Heads: {num_heads}")
        print(f"  - Head Dimension: {head_dim}")
        print(f"  - GQA Groups: {gqa_groups}")
        print(f"  - Attention Mask: {attn_mask_type}")
        print(f"  - Window Size: {window_size}")
        print(f"  - Context Parallel: {use_cp}")
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

    # Run benchmarks
    for dtype in args.dtypes:
        for qkv_format in args.qkv_formats:
            for seq_len in args.seq_lens:
                for batch_size in args.batch_sizes:
                    for use_cp in args.context_parallel:
                        # Skip CP if world_size is 1
                        if use_cp and dist.get_world_size() == 1:
                            continue

                        for ws in args.window_sizes:
                            for use_ulysses in args.use_ulysses:
                                for head_first in args.head_first:
                                    metric = benchmark_zigzag(
                                        dtype=dtype,
                                        qkv_format=qkv_format,
                                        batch_size=batch_size,
                                        seq_len=seq_len,
                                        num_heads=args.num_heads,
                                        head_dim=args.head_dim,
                                        gqa_groups=args.gqa_groups,
                                        attn_mask_type=args.attn_mask_type,
                                        window_size=ws,
                                        warmup=args.warmup,
                                        iterations=args.iterations,
                                        use_cp=use_cp,
                                        use_ulysses=use_ulysses,
                                        head_first=head_first,
                                        profile=args.profile
                                    )

                            # Save results
                            result = {
                                "dtype": dtype,
                                "qkv_format": qkv_format,
                                "batch_size": batch_size,
                                "seq_len": seq_len,
                                "num_heads": args.num_heads,
                                "head_dim": args.head_dim,
                                "context_parallel": use_cp,
                                "window_size": ws,
                                "world_size": dist.get_world_size(),
                                **metric,
                            }
                            results.append(result)

    # Print summary table if rank 0
    if dist.get_rank() == 0:
        print("\nBenchmark Summary:")
        print(f"{'=' * 120}")
        header = f"{'Dtype':^8} | {'QKV Format':^8} | {'B':^4} | {'Seq Len':^7} | {'CP':^5} | {'Window':^7} | {'FWD (ms)':^10} | {'FWD+BWD (ms)':^12} | {'Memory (MB)':^12} | {'TFLOPs/s':^10}"
        print(header)
        print(f"{'-' * 120}")

        for result in results:
            row = (
                f"{result['dtype']:^8} | "
                f"{result['qkv_format']:^8} | "
                f"{result['batch_size']:^4} | "
                f"{result['seq_len']:^7} | "
                f"{str(result['context_parallel']):^5} | "
                f"{str(result['window_size']):^7} | "
                f"{result['forward_time_ms']:^10.2f} | "
                f"{result['fwd_bwd_time_ms']:^12.2f} | "
                f"{result['memory_mb']:^12.2f} | "
                f"{result['tflops']:^10.2f}"
            )
            print(row)

        print(f"{'=' * 120}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ZigZagRingFlashAttnFunc")

    # Model parameters
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["bf16"],
        choices=["fp16", "bf16", "fp32"],
        help="Data types to benchmark",
    )
    parser.add_argument(
        "--qkv-formats",
        nargs="+",
        default=["bshd"],
        choices=["bshd", "sbhd"],
        help="QKV formats to benchmark",
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
        "--num-heads", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--head-dim", type=int, default=64, help="Dimension of each head"
    )
    parser.add_argument(
        "--gqa-groups",
        type=int,
        default=None,
        help="Number of GQA groups (defaults to num_heads if None)",
    )
    parser.add_argument(
        "--attn-mask-type",
        default="causal",
        choices=["causal", "no_mask"],
        help="Attention mask type",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=[None],
        help="Window sizes for sliding window attention",
    )

    # Parallelism parameters
    parser.add_argument(
        "--context-parallel",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[False, True],
        help="Whether to use context parallelism",
    )
    parser.add_argument(
        "--use-ulysses",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[False, True],
        help="Whether to use context parallelism",
    )
    parser.add_argument(
        "--head-first",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[False, True],
        help="Whether to use context parallelism",
    )
    # Benchmark parameters
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling and save the results to the specified path"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_benchmarks(args)
