export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=8 test_attn.py
