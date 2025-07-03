echo "### Megatron CP RingAttn"
bash ./benchmark_dpa.sh
echo "### Megatron Ulysses"
bash ./benchmark_ds_ulysses.sh
echo "### DoubleRing"
bash ./bench_loong.sh
echo "### BurstAttention"
bash ./bench_burst.sh"
echo "### USP"
bash ./bench_usp.sh
