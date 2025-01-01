
#!/bin/bash

# sizes=( 524288 )
# sizes=( 65536 131072 262144 524288 )
sizes=( 2097152 )

export NODES="g41 g47 g48 g71"
export NODES="g47 g43 g74 g49"
export NODES="g41 g43 g45 g47"

DOCKER_DIR=/workspace/workspace/Megatron-DeepSpeed
export WORLD_SIZE=4
export LOG_FILE=$DOCKER_DIR/7b_2048k_vocab120k.log
export MODEL="7b"
echo $LOG_FILE
touch $LOG_FILE
sizes=( 2097152 )
for size in ${sizes[@]}; do
  for cp in 32; do
    if [[ $size -eq 262144 && $cp -eq 16 ]]; then
      continue
    fi
    if [[ $size -eq 524288 && ($cp -eq 8 || $cp -eq 16) ]]; then
      continue
    fi
    export CP_SIZE=$cp
    echo "Running size $size with method $method" >> summary.txt
    bash submit.sh "bash build_run.sh $size" 
    # bash submit.sh "bash conda.sh $size $method" 
  done
done
