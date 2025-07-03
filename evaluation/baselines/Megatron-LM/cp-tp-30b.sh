#!/bin/bash

# sizes=( 524288 )

sizes=( 32768 )
methods=(
  # "burst_double"
  "megatron-cp"
  # "burst_ulysses"
  # "burst_double"
)
export CP_SIZE=8
export NODES="g41 g43 g47 g49 g71 g72 g73 g74 g75"
export NODES="g41 g43 g47 g49"
export NODES="g41 "
DOCKER_DIR=Megatron-LM
export MODEL="2.7b"
export WORLD_SIZE=1
export TP_SIZE=1
export LOG_FILE=$DOCKER_DIR/30btp-cp.log
echo $LOG_FILE
for method in ${methods[@]}; do
  echo "Running method $method" >> summary.txt
  for size in ${sizes[@]}; do
      echo "Running size $size with method $method" >> summary.txt
      bash submit.sh "bash build_run.sh $size $method" 
      # bash submit.sh "bash conda.sh $size $method" 
  done

done


