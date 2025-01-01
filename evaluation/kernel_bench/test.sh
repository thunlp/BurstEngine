
cp ./attn.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py
torchrun --nproc_per_node 8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:7778 test_te_attn.py  --cp_comm_type=a2a
