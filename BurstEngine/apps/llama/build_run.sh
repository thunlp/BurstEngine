WORK_DIR=/BurstEngine/apps/llama

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
  -e MASTER_ADDR=$MASTER_ADDR \
  -e seqlen=$1 \
  -e method=$2 \
  -e PROFILE=$PROFILE \
  -e CP_SIZE=$CP_SIZE \
  -e ABLATION=$ABLATION \
  -e WORLD_SIZE=$WORLD_SIZE \
  -e MASTER_PORT=6000 \
  -e MODEL=$MODEL \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e MODEL=$MODEL \
  -e IS_WORKER=$IS_WORKER \
  -e LOG_FILE=$LOG_FILE \
  -e CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS \
  -e UCX_NET_DEVICES=$UCX_NET_DEVICES \
  -e GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
  -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -v $PROJECT_DIR/BurstEngine:/BurstEngine \
  --ulimit memlock=-1 \
  --ulimit stack=67108864  \
  --privileged=true \
  --name burst_engine \
  burst_engine:latest /bin/bash -c "source /env.sh && cd $WORK_DIR && bash ./multi.sh $1 $2"
