BLACK_LIST=("g70")
AVAIL=`bash ${PROJECT_DIR}/check.sh -l `
BLACK_LIST_REGEX=$(IFS="|"; echo "${BLACK_LIST[*]}")
AVAIL=$(echo "$AVAIL" | grep -Ev "$BLACK_LIST_REGEX")

NODE_COUNT=$(echo "$AVAIL" | wc -l | tr -d ' ')
if [ -z "$AVAIL" ] || [ "$NODE_COUNT" -lt "$WORLD_SIZE" ]; then
  echo "No available nodes"
  exit 1
else
  if [ -z "$NODES" ]; then
    echo "Available nodes: " &&
    NODES=$(echo $AVAIL|tr " " "\n" |tail -n $WORLD_SIZE|tr "\n" " " )
  fi
  echo "Used Nodes $NODES"
  export MASTER_ADDR=`ifconfig $NCCL_SOCKET_IFNAME|grep inet|awk '{print $2}'|head -n 1`
  echo "MASTER_ADDR: $MASTER_ADDR"
  echo "WORLD_SIZE: $WORLD_SIZE"
  source $PROJECT_DIR/env.sh
  # pdsh -R ssh -w "$NODES" "docker stop $(docker ps -a -q)"
  pdsh -R ssh -w "$NODES" "export PROJECT_DIR=$PROJECT_DIR; export LOG_FILE=$LOG_FILE;export MASTER_ADDR=$MASTER_ADDR;export WORLD_SIZE=$WORLD_SIZE; export CP_SIZE=$CP_SIZE; export MODEL=$MODEL;export TP_SIZE=$TP_SIZE;export CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS; export UCX_NET_DEVICES=$UCX_NET_DEVICES; export GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME; export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME; export NCCL_IB_HCA=$NCCL_IB_HCA; cd `pwd` && $1"
  # pdsh -R ssh -w "$NODES" "bash /home/test/test01/sa/kill.sh"

fi
