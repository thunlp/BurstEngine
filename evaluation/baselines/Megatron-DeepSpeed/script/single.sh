

#!/bin/bash

# sizes=( 524288 )
sizes=( 131072 262144 524288 )
sizes=(  65536 131072 262144 524288 )

export CP_SIZE=8
export NODES="g71"
DOCKER_DIR=/workspace/workspace/Megatron-DeepSpeed
export WORLD_SIZE=1
export LOG_FILE=$DOCKER_DIR/single_ulysses_13b.log
export MODEL="7b"
echo $LOG_FILE
for cp in 2 4 8; do
  export CP_SIZE=$cp
  for size in ${sizes[@]}; do
      # echo "Running size $size with method $method" >> summary.txt
      # bash submit.sh "bash build_run.sh $size" 
      bash ulysses.sh $size $cp
  done
done
