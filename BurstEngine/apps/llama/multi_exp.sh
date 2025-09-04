#!/bin/bash

# sizes=( 1048576 2097152 )
# sizes = ( 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 )
sizes=( 131072 )
methods=(
  # "burst_sl_loss_sl_ckpt"
  "burst"
  # "burst_sl_loss_whole_ckpt_sliding"
  # "burst_whole_ckpt"
  # "burst_sl_loss"
  # "burst_whole_ckpt_sl_loss"
  # "burst_sl_ckpt_sl_loss"
  # "burst_sl_loss"
  # "burst_ulysses_sl_loss"
  # "burst_sl_loss_sl_ckpt"
  # "burst_ulysses_sl_loss_sl_ckpt"
  # "burst_ulysses_sl_loss"
  # "burst_ulysses_whole_ckpt_sl_loss"
  # "burst_sl_ckpt"
  # "burst_whole_ckpt"
  # "burst_ulysses"
  # "burst_ulysses_whole_ckpt"
  # "burst_ulysses_sl_ckpt"
  # "burst_ulysses"
  # "burst_whole_ckpt"
  # "burst_sl_ckpt"
)
ablation=(
  "false"
)
export PROFILE="false"
export NODES="bjdx1 bjdx2 bjdx3 bjdx4"
export MODEL="7b"
# export NODES="g43 g47 g48 g49"
DOCKER_DIR=/BurstEngine/apps/llama
export LOG_FILE=$DOCKER_DIR/sl_ckpt_exp.log
echo $LOG_FILE
for method in ${methods[@]}; do
  for ablation in ${ablation[@]}; do
  echo "Running method $method" >> summary.txt
  for size in ${sizes[@]}; do
    for cp in 32; do
      export WORLD_SIZE=$((cp > 8 ? cp / 8 : 1))
      size=$((32768 * cp))
      # if [[ $size -eq 262144 && $cp -eq 8 ]]; then
      #   continue
      # fi
      # if [[ $size -eq 524288 && ($cp -eq 8 || $cp -eq 16) ]]; then
      #   continue
      # fi
      export CP_SIZE=$cp
      export ABLATION=$ablation
      echo "Running size $size with method $method" >> summary.txt
      # export LOG_FILE=$LOG_FILE;export MASTER_ADDR=g41; export MODEL=$MODEL;
      # bash submit_slurm.sh "bash build_run.sh $size $method " 
      bash ./submit.sh "bash build_run.sh $size $method " 
      # bash multi.sh $size $method $cp
      done
    done
  done
done

