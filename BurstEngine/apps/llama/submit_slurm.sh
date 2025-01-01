
BLACK_LIST=("g70")
AVAIL=`bash /home/test/test01/sa/check.sh -l `
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
  MASTER_ADDR=`echo $NODES | cut -d ' ' -f 1`
  echo "Used Nodes $NODES"
  echo "WORLD_SIZE: $WORLD_SIZE"
  echo $NODES
  sbatch --nodes=$WORLD_SIZE -w "$NODES" ./slurm/srun.sh $1

fi
