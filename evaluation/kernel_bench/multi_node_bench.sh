#!/bin/bash
export NODES="g43 g45 g69 g75"
WORLD_SIZE=$(echo $NODES | wc -w)

# Function to handle job submission
submit_job() {
  MASTER_ADDR=$(echo "$NODES" | cut -d ' ' -f 1)

  echo "Used Nodes: $NODES"
  echo "MASTER_ADDR: $MASTER_ADDR"
  echo "WORLD_SIZE: $WORLD_SIZE"
  pdsh -R ssh -w "$NODES" "bash /home/test/test01/sa/kill.sh"
  pdsh -R ssh -w "$NODES" "docker stop \$(docker ps -a -q)"
  pdsh -R ssh -w "$NODES" "
    export MASTER_ADDR=$MASTER_ADDR;
    export WORLD_SIZE=$WORLD_SIZE;
    cd $(pwd) && bash docker_entry.sh $MASTER_ADDR $WORLD_SIZE
  "
  pdsh -R ssh -w "$NODES" "bash /home/test/test01/sa/kill.sh"
  pdsh -R ssh -w "$NODES" "docker stop \$(docker ps -a -q)"
}

# Main script loop
echo $LOG_FILE > summary.txt

echo "Running size $size with method $method" >> summary.txt
submit_job 
