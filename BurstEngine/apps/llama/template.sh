#!/bin/bash

sizes=( 1048576 )
methods=(
  "burst_sl_loss_sl_ckpt"
  "burst_whole_ckpt_sl_loss"
)
ablation=(
  "false"
)
export PROFILE="false"
export NODES="g45 g43 g72 g73"
export NODES="g49 g41 g47 g73"
export MODEL="13b"
for method in ${methods[@]}; do
  for ablation in ${ablation[@]}; do
  echo "Running method $method" >> summary.txt
  for size in ${sizes[@]}; do
    for cp in 8; do
      export WORLD_SIZE=$((cp > 8 ? cp / 8 : 1))
      size=$((32768 * cp)) # per gpu have 32k data
      export CP_SIZE=$cp
      export ABLATION=$ablation
      echo "Running size $size with method $method" >> summary.txt
      bash submit.sh "bash build_run.sh $size $method " 
      done
    done
  done
done

