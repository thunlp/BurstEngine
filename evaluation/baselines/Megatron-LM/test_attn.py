import torch
import time
from transformer_engine.pytorch.attention import attn_forward_func_with_cp
import os
from burst_attn import burst_attn_func

# Setting up torch device
torch.distributed.init_process_group(backend="nccl", init_method="env://")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(torch.distributed.get_rank())
sub_cp_group = torch.distributed.new_group(ranks=list(range(torch.distributed.get_world_size())))
cp_global_ranks = torch.distributed.get_process_group_ranks(sub_cp_group)
# cp_stream = torch.cuda.current_stream()
cp_stream = torch.cuda.Stream(-1)
scale_stream = torch.cuda.current_stream()
world_size = torch.distributed.get_world_size()
_print = print
def print_func(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        _print(*args, **kwargs)
print = print_func
def sync():
    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(cp_stream)
def start_event():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event

def end_event(start_event):
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize()  # wait for the event to finish
    return start_event.elapsed_time(end)  # returns the elapsed time in milliseconds
# cp_global_ranks = torch.distributed.get_process_group_ranks(None)

# Function to calculate the number of floating-point operations (specific to the attention mechanism)
# Mock function (replace with the actual function if available)
# Configurations

# Generate random data for the inputs
def Benchmark(batch_size, seq_len, num_heads, hidden_dim):
    q = torch.randn(batch_size, seq_len, num_heads, hidden_dim, device=device, dtype= torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, hidden_dim, device=device, dtype= torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, hidden_dim, device=device, dtype= torch.float16)
    offsets = [0]
    for i in range(batch_size):
        offsets.append(offsets[-1] + seq_len)
    cu_seqlens_q = torch.tensor(offsets, device="cuda", dtype=torch.int)  # Example format, specify correctly
    cu_seqlens_k = torch.tensor(offsets, device="cuda", dtype=torch.int)  # Example format, specify correctly
    max_seqlen_q = seq_len
    max_seqlen_k = seq_len
    cu_seqlens_q_padded = torch.tensor([0, seq_len-1], device="cuda", dtype=torch.int)
    cu_seqlens_kv_padded = torch.tensor([0, seq_len-1], device="cuda", dtype=torch.long)
    dropout_p = 0
    softmax_scale = None

# Warm-up

    attn_mask_type="causal"
    def calculate_attention_flops(batch_size, seq_len, hidden_size, num_attention_heads):

        scale = 0.5 if attn_mask_type == "causal" else 1.0
        fs = 4 * batch_size * seq_len * seq_len * hidden_size * scale

        return fs
    warm_up = True
    if warm_up:
        for _ in range(10):
            _ = attn_forward_func_with_cp(is_training=True, q=q, k=k, v=v, 
                                          cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                          max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                                          cu_seqlens_q_padded=cu_seqlens_q_padded, cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                          dropout_p=dropout_p, softmax_scale=softmax_scale,
                                          cp_group=None, cp_global_ranks=cp_global_ranks, cp_stream=cp_stream)

# Benchmark
    num_iterations = 100
    os.environ["NVTE_BATCH_MHA_P2P_COMM"] = "0"
#     sync()
#     start_time = time.time()
#
#     for _ in range(num_iterations):
#         attn_forward_func_with_cp(is_training=True, q=q, k=k, v=v, 
#                                   cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
#                                   max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
#                                   cu_seqlens_q_padded=cu_seqlens_q_padded, cu_seqlens_kv_padded=cu_seqlens_kv_padded,
#                                   dropout_p=dropout_p, softmax_scale=softmax_scale,
#                                   cp_group=None, cp_global_ranks=cp_global_ranks, cp_stream=cp_stream)
#     sync()
#     total_time = time.time() - start_time
#     total_flops = calculate_attention_flops(batch_size, seq_len*world_size, num_heads * hidden_dim, num_heads) * num_iterations / world_size
#
# # TFLOPs calculation
#     tflops = (total_flops / total_time) / 1e12
#     print(f"TE Total TFLOPs: {tflops:.2f}")

    sync()
    start_event1 = start_event()
    os.environ["NVTE_BATCH_MHA_P2P_COMM"] = "1"
    # with torch.cuda.nvtx.range("transformer_engine_func_range"):
    for _ in range(num_iterations):
        attn_forward_func_with_cp(is_training=True, q=q, k=k, v=v, 
                                  cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                  max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                                  attn_mask_type=attn_mask_type,
                                  cu_seqlens_q_padded=cu_seqlens_q_padded, cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                  dropout_p=dropout_p, softmax_scale=softmax_scale,
                                  cp_group=None, cp_global_ranks=cp_global_ranks, cp_stream=cp_stream)
    total_time  = end_event(start_event1) / 1000
    total_flops = calculate_attention_flops(batch_size, seq_len*world_size, num_heads * hidden_dim, num_heads) * num_iterations / world_size

# TFLOPs calculation
    tflops = (total_flops / total_time) / 1e12
    print(f"TE Batch Total TFLOPs: {tflops:.2f}")
    sync()
    start_event2 = start_event()
    with torch.cuda.nvtx.range("burst_attn_func_range"):
        for _ in range(num_iterations):
            burst_attn_func(q, k, v, None, "cuda", attn_mask_type=="causal", True, False, None, [None, None], scale_stream)
    total_time  = end_event(start_event2) / 1000
    total_flops = calculate_attention_flops(batch_size, seq_len*world_size, num_heads * hidden_dim, num_heads) * num_iterations / world_size

# TFLOPs calculation
    tflops = (total_flops / total_time) / 1e12

    print(f"Burst Total TFLOPs: {tflops:.2f}")
    sync()

total_tok = 4096 * 8
for seqlen in [16384, 8192, 4096]:
    print(f"Total SeqLen: {seqlen * world_size}")
    Benchmark(total_tok // seqlen, seqlen, 32, 128)
    print("***********************************")
