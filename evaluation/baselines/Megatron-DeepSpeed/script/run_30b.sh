#!/bin/bash

# sizes=( 524288 )
# sizes=( 65536 131072 262144 524288 )
sizes=( 65536 )

export NODES="g41 g43 g47 g49 g71 g72 g73 g74 g75"
 
DOCKER_DIR=Megatron-DeepSpeed
export WORLD_SIZE=8
export LOG_FILE=$DOCKER_DIR/30b-64k.log
export LOG_FILE=$DOCKER_DIR/test-64k.log
export MODEL="30b"
echo $LOG_FILE
# sizes=( 65536 131072 262144 524288 )
for size in ${sizes[@]}; do
  for cp in 64; do
    export CP_SIZE=$cp
    echo "Running size $size with method $method" >> summary.txt
    bash submit.sh "bash build_run.sh $size" 
    # bash submit.sh "bash conda.sh $size $method" 
  done
done
