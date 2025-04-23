WORK_DIR=/home/test/test01/sa/workspace/burst_engine/evaluation/kernel_bench/

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
# echo $WORLD_SIZE
benchmark_script="bench_usp.sh"
log_name="${benchmark_script%.sh}"
TIME=$(date +%Y_%m_%d_%H_%M_%S)
echo $log_name
if [[ -z "$IS_WORKER" || "$IS_WORKER" != "1" ]]; then
  LOG_PATH=$WORK_DIR/logs/${log_name}-${TIME}.log
else
  LOG_PATH=/tmp/${log_name}-${TIME}.log
  echo $LOG_PATH
fi
docker run ${run_arg} --log-driver=json-file -m 500G --rm  -u root --ipc=host \
  --shm-size="32g" \
  --net=host \
  --gpus all --ipc=host \
  -v /home/test/test01/:/home/test/test01/  \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e MASTER_PORT=6000 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864  \
  --privileged=true \
  4b4311a4be04750d95789fb0ba4c5 /bin/bash -c "cd $WORK_DIR && bash ./$benchmark_script $1 $2 2>&1| tee $LOG_PATH "

