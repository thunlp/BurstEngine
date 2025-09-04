#!/bin/bash
###
#SBATCH --job-name=nccl_test
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=20:00
#SBATCH --output="%x.out"
#SBATCH --exclusive

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

export NCCL_SOCKET_IFNAME=eth0
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1

# Dynamic Connections can be forced as transport
export UCX_TLS=dc,self

# Enable network collections
export NCCL_COLLNET_ENABLE=1

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

srun --container-image=kj \
     --container-remap-root --no-container-mount-home \
     /opt/nccl_tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 1
