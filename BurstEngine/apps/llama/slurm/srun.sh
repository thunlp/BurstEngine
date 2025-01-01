#!/bin/bash
#SBATCH --account=test
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
cd ..
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=26789
cmd="bash $1"
echo $cmd
echo $MASTER_ADDR
echo $MASTER_PORT
echo $SLURM_JOB_NODELIST
# $cmd

