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
ori_pwd=`pwd`
# cp -r ../../Burst-Attention/  /root/Burst-Attention && cd /root/Burst-Attention && pip install .
export NCCL_IB_QPS_PER_CONNECTION=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
cd $ori_pwd
cmd="torchrun --nnodes $world_size --nproc_per_node 8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$host:7778 benchmark_burst.py"
echo $cmd
$cmd
