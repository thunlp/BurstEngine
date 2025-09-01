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
  --ulimit memlock=-1 \
  --ulimit stack=67108864  \
  --privileged=true \
  burst_engine:latest /bin/bash -c "source $PROJECT_DIR/env.sh && cd $WORK_DIR && bash ./multi.sh $1 $2"
