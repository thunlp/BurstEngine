
HOSTNAME=`hostname`
MASTER_ADDR=localhost



run_arg="-itd"
# docker stop $(docker ps -a -q)
# docker rmi burst_run:latest
# docker bbash uild -t burst_run:latest .
docker run ${run_arg} --log-driver=json-file -m 500G --rm  -u root --ipc=host \
  --shm-size="32g" \
  --net=host \
  --gpus all --ipc=host \
  -v /home/test/test01/sa/workspace:/workspace/workspace  \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e seqlen=$1 \
  -e method=$2 \
  -e CP_SIZE=$CP_SIZE \
  -e WORLD_SIZE=$WORLD_SIZE \
  -e MASTER_PORT=6000 \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e IS_WORKER=$IS_WORKER \
  -e NVTE_DEBUG_LEVEL=0 \
  -e LOG_FILE=$LOG_FILE \
  --ulimit memlock=-1 \
  --privileged=true \
  --ulimit stack=67108864  \
  78283aaa4c6345ad05327e6093eb83c87 /bin/bash 
