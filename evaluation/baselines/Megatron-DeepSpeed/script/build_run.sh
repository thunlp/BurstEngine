
HOSTNAME=`hostname`
if [ -z $MASTER_ADDR ]; then
  MASTER_ADDR=localhost
fi

if [ "$HOSTNAME" != "$MASTER_ADDR" ]; then
  IS_WORKER=1
fi


if [ -z $IS_WORKER ]; then
  run_arg="-i"
else
  run_arg="-i"
fi
docker stop $(docker ps -a -q)
# docker rmi burst_run:latest
# docker bbash uild -t burst_run:latest .
docker run ${run_arg} --log-driver=json-file -m 500G --rm  -u root --ipc=host \
  --shm-size="32g" \
  --net=host \
  --gpus all --ipc=host \
  -v /home/test/test01/sa/workspace:/workspace/workspace  \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e seqlen=$1 \
  -e MODEL=$MODEL \
  -e CP_SIZE=$CP_SIZE \
  -e WORLD_SIZE=$WORLD_SIZE \
  -e MASTER_PORT=6000 \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e IS_WORKER=$IS_WORKER \
  -e LOG_FILE=$LOG_FILE \
  -e CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS \
  -e UCX_NET_DEVICES=$UCX_NET_DEVICES \
  -e GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
  -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  --ulimit memlock=-1 \
  --ulimit stack=67108864  \
  -v /shared/sc_workspace/BE/evaluation/baselines/Megatron-DeepSpeed:/Megatron-DeepSpeed \
  -v $PROJECT_DIR/data:/data \
  --privileged=true \
  burst_engine:latest /bin/bash -c " \
  cd /Megatron-DeepSpeed/script \
 && source /env.sh && bash ./ulysses.sh"
