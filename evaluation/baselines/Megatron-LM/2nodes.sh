
#!/bin/bash

sizes=(262144)
methods=(
  "ulysses"
)
export CP_SIZE=16

DOCKER_DIR=/workspace/workspace/Megatron-LM
export WORLD_SIZE=2
export LOG_FILE=$DOCKER_DIR/16gpu_burst_ulysses_256k.log
echo $LOG_FILE
for method in ${methods[@]}; do
  echo "Running method $method" >> summary.txt
  for size in ${sizes[@]}; do
      echo "Running size $size with method $method" >> summary.txt
      bash submit.sh "bash build_run.sh $size $method" 
      # bash submit.sh "bash conda.sh $size $method" 

  done

done
