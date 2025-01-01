torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=g3010 benchmark.py

