#!/bin/bash

sizes=( 32768 )
# sizes=( 131072 262144 )

# sizes=( 524288  )
methods=(
  # "burst_double"
  "megatron-cp"
  # "burst_ulysses"
  # "burst_double"
)
export CP_SIZE=4
export TP_SIZE=8
export NODES="bjdx1 bjdx2 bjdx3 bjdx4"
DOCKER_DIR=Megatron-LM
export MODEL="7b"
export WORLD_SIZE=4
export LOG_FILE=$DOCKER_DIR/baseline-13b.log
echo $LOG_FILE
for method in ${methods[@]}; do
  echo "Running method $method" >> summary.txt
  for size in ${sizes[@]}; do
      echo "Running size $size with method $method" >> summary.txt
      bash submit.sh "bash build_run.sh $size $method" 
      # bash $PROJECT_DIR/submit.sh "bash conda.sh $size $method" 

  done

done
