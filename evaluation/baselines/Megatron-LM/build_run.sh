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
  -e method=$2 \
  -e TP_SIZE=$TP_SIZE \
  -e CP_SIZE=$CP_SIZE \
  -e WORLD_SIZE=$WORLD_SIZE \
  -e MASTER_PORT=6238 \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e IS_WORKER=$IS_WORKER \
  -e LOG_FILE=$LOG_FILE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864  \
  --privileged=true \
  78283aaa4c6345ad05327e6093eb83c87 /bin/bash -c " \
  cd Megatron-LM \
    && cp ./patch.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py \
  && bash ./multi.sh"
