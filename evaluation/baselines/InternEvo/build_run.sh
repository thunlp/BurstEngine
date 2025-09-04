
HOSTNAME=`hostname`
if [ -z $MASTER_ADDR ]; then
  MASTER_ADDR=localhost
fi

if [ "$HOSTNAME" != "$MASTER_ADDR" ]; then
  IS_WORKER=1
fi


if [ -z $IS_WORKER ]; then
  run_arg="-i"
  running_log=13b_64x_running.log
else
  run_arg="-i"
  running_log="/tmp/loong_tmp.log"
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
  -e WORLD_SIZE=$WORLD_SIZE \
  -e LOG_FILE=$LOG_FILE \
  -e MASTER_PORT=6000 \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e IS_WORKER=$IS_WORKER \
  --ulimit memlock=-1 \
  --ulimit stack=67108864  \
  --privileged=true \
  burst_engine:latest /bin/bash -c " \
  cd InternEvo \
  && bash pre.sh \
  && mkdir -p "$(dirname "$running_log")" \
  && echo 'hp $HP_SIZE cp $CP_SIZE seqlen $1 with ckpt $sele_ckpt' >> $running_log \
  && touch $running_log \
  && source /env.sh \
  && bash exp.sh 2>&1 | tee -a $running_log"
