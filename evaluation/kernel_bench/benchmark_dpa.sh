
#!/bin/bash

# Example benchmark script for DotProductAttention

# Basic single GPU benchmark
# echo "Running single GPU benchmark..."
# python3 benchmark_dpa.py \
#     --dtypes bf16 \
#     --qkv-formats bshd \
#     --kernels FlashAttention \
#     --batch-sizes 32 \
#     --seq-lens 1024 2048 4096 \
#     --num-heads 16 \
#     --head-dim 64 \
#     --warmup 3 \
#     --iterations 10

# Multi-GPU benchmark with context parallelism
if [ -z "$1" ]; then
  host="localhost"
else
  host=$1
fi
if [ -z "$2" ]; then
  world_size=1
else
  world_size=$2
fi

echo "Running multi-GPU benchmark with RingAttention..."
# torchrun --rdzv_backend=c10d --rdzv_id=1 --nnodes $world_size --rdzv_endpoint=$host:7778 --nproc_per_node=8 benchmark_dpa.py \
#     --dtypes bf16 \
#     --qkv-formats bshd \
#     --kernels FlashAttention \
#     --batch-sizes 1 \
#     --seq-lens 32768 65536 131072 \
#     --num-heads 32 \
#     --head-dim 128 \
#     --attn-mask-type causal \
#     --context-parallel True \
#     --cp-comm-type p2p \
#     --warmup 3 \
#     --iterations 10
cp ./attn.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py
# echo "Running multi-GPU benchmark with Ring Attention..."
# torchrun --rdzv_backend=c10d --rdzv_id=1 --nnodes $world_size --rdzv_endpoint=$host:7778 --nproc_per_node=8 benchmark_dpa.py \
#     --dtypes bf16 \
#     --qkv-formats bshd \
#     --kernels FlashAttention \
#     --batch-sizes 1 \
#     --seq-lens 131072 262144 524288 \
#     --num-heads 32 \
#     --head-dim 128 \
#     --attn-mask-type causal \
#     --context-parallel True \
#     --cp-comm-type p2p \
#     --warmup 3 \
#     --iterations 10 \
#
echo "Running multi-GPU benchmark with Attention..."
export CUDA_DEVICE_MAX_CONNECTIONS=1
        # --seq-lens 1048576 2097152 4194304 \
for op in p2p; do
    torchrun --rdzv_backend=c10d --rdzv_id=1 --nnodes $world_size --rdzv_endpoint=$host:7778 --nproc_per_node=8 benchmark_dpa.py \
        --dtypes bf16 \
        --qkv-formats bshd \
        --kernels FlashAttention \
        --batch-sizes 1 \
        --num-heads 40 \
        --head-dim 128 \
        --seq-lens 524288 \
        --attn-mask-type causal \
        --context-parallel True \
        --cp-comm-type $op \
        --iterations 20
done
#
# echo "Running multi-GPU benchmark with All2All RingAttention (USP)..."
# torchrun --rdzv_backend=c10d --rdzv_id=1 --nnodes $world_size --rdzv_endpoint=$host:7778 --nproc_per_node=8 benchmark_dpa.py \
#     --dtypes bf16 \
#     --qkv-formats bshd \
#     --kernels FlashAttention \
#     --batch-sizes 1 \
#     --seq-lens 131072 262144 524288 \
#     --num-heads 32 \
#     --head-dim 128 \
#     --attn-mask-type causal \
#     --context-parallel True \
#     --cp-comm-type a2a+p2p \
#     --warmup 3 \
#     --iterations 10

# # GQA benchmark
# echo "Running GQA benchmark..."
# python3 benchmark_dpa.py \
#     --dtypes bf16 \
#     --qkv-formats bshd \
#     --kernels FlashAttention \
#     --batch-sizes 32 \
#     --seq-lens 4096 \
#     --num-heads 16 \
#     --head-dim 64 \
#     --gqa-groups 4 \
#     --warmup 3 \
#     --iterations 10
#
# # FP8 benchmark
# echo "Running FP8 benchmark..."
# python3 benchmark_dpa.py \
#     --dtypes fp8 \
#     --qkv-formats bshd \
#     --kernels FlashAttention \
#     --batch-sizes 32 \
#     --seq-lens 4096 \
#     --num-heads 16 \
#     --head-dim 64 \
#     --fp8-mha \
#     --warmup 3 \
#     --iterations 10
#
# # Comprehensive benchmark
# echo "Running comprehensive benchmark..."
# python3 benchmark_dpa.py \
#     --dtypes bf16 fp16 \
#     --qkv-formats bshd thd \
#     --kernels FlashAttention FusedAttention \
#     --batch-sizes 16 32 \
#     --seq-lens 1024 4096 \
#     --num-heads 16 \
#     --head-dim 64 \
#     --warmup 3 \
#     --iterations 10
#
# echo "All benchmarks completed!"
