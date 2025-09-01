#!/bin/bash

# sizes=( 524288 )
sizes=( 131072 262144 524288 1048576 )
sizes=( 65536 131072 262144 524288)
# sizes=( 32768 65536 )

methods=(
  "megatron-cp"
)
export NODES="g41"
DOCKER_DIR=Megatron-LM
export WORLD_SIZE=1
export LOG_FILE=$DOCKER_DIR/single_node_cp_13b.log
export MODEL="7b"
echo $LOG_FILE
for method in ${methods[@]}; do
  echo "Running method $method" >> summary.txt
  for size in ${sizes[@]}; do
      echo "Running size $size with method $method" >> summary.txt
      # bash $PROJECT_DIR/submit.sh "bash build_run.sh $size $method" 
    for cp in 2 4 8; do 
      export CP_SIZE=$cp
      bash multi.sh $size $method
    done
      # bash $PROJECT_DIR/submit.sh "bash conda.sh $size $method" 
  done
done
