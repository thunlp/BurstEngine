#!/bin/bash

# sizes=( 524288 )
sizes=( 131072 262144 )

sizes=( 2097152 )
methods=(
  # "burst_double"
  "megatron-cp"
  # "burst_ulysses"
  # "burst_double"
)
export CP_SIZE=32
export TP_SIZE=1
export NODES="g45 g43 g72 g73 g41 g47 g48 g74"
DOCKER_DIR=Megatron-LM
export MODEL="13b"
export WORLD_SIZE=4
export LOG_FILE=$DOCKER_DIR/baseline-13b.log
echo $LOG_FILE
for method in ${methods[@]}; do
  echo "Running method $method" >> summary.txt
  for size in ${sizes[@]}; do
      echo "Running size $size with method $method" >> summary.txt
      bash submit.sh "bash build_run.sh $size $method" 
      # bash submit.sh "bash conda.sh $size $method" 

  done

done
