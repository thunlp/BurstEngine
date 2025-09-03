BLACK_LIST=("g70")
AVAIL=`bash ${PROJECT_DIR}/check.sh -l `
BLACK_LIST_REGEX=$(IFS="|"; echo "${BLACK_LIST[*]}")
AVAIL=$(echo "$AVAIL" | grep -Ev "$BLACK_LIST_REGEX")

NODE_COUNT=$(echo "$AVAIL" | wc -l | tr -d ' ')
if [ -z "$AVAIL" ] || [ "$NODE_COUNT" -lt "$WORLD_SIZE" ] && [ -z "$NODES" ]; then
  echo "No available nodes"
  exit 1
else
  if [ -z "$NODES" ]; then
    echo "Available nodes: " &&
    NODES=$(echo $AVAIL|tr " " "\n" |tail -n $WORLD_SIZE|tr "\n" " " )
  fi
  # MASTER_ADDR=`echo $NODES | cut -d ' ' -f 1`
  MASTER_ADDR=`ifconfig $NCCL_SOCKET_IFNAME|grep inet|awk '{print $2}'|head -n 1`
  echo "Used Nodes $NODES"
  echo "MASTER_ADDR: $MASTER_ADDR"
  echo "WORLD_SIZE: $WORLD_SIZE"
  echo $MODEL
  pdsh -R ssh -w "$NODES" "export LOG_FILE=$LOG_FILE;export MASTER_ADDR=$MASTER_ADDR;export WORLD_SIZE=$WORLD_SIZE; export CP_SIZE=$CP_SIZE; export MODEL=$MODEL; export PROFILE=$PROFILE; export ABLATION=$ABLATION; cd `pwd` && $1"
  pdsh -R ssh -w "$NODES" "bash /home/test/test01/sa/kill.sh"
  pdsh -R ssh -w "$NODES" "docker stop $(docker ps -a -q)"

fi
