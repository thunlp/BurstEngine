# BurstEngine: An Efficient Distributed Framework for Training Transformers on Extremely Long Sequences

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](http://arxiv.org/abs/2509.19836)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

BurstEngine is a highly efficient distributed framework specifically designed for training transformer-based large language models (LLMs) and large multimodal models (LMMs) on extremely long sequences of over 1M tokens. It addresses the critical challenges of quadratic time and space complexities in attention mechanisms while achieving superior performance compared to existing state-of-the-art methods.

## 🚀 Key Features

### Core Optimizations

- **BurstAttention**: A highly optimized distributed attention implementation with:
  - **Backward Communication Optimization**: Reduces ~25% communication overhead compared to RingAttention
  - **Topology-aware Ring Communication**: Maximizes network bandwidth utilization across intra-node and inter-node connections
  - **Fine-grained Communication-Computation Overlap**: Minimizes overall communication overhead through specialized double buffer design

- **Sequence-Level Selective Checkpointing**: Optimizes the trade-off between memory overhead and computation overhead at the sequence level

- **Sequence-Level Fusion of LM Head and Loss Function**: Reduces memory overhead of storing intermediate states in language modeling head

- **Sparse Attention Integration**: Supports various sparse attention patterns including causal masking, sliding-window masking, and block-wise sparse patterns

### Performance Achievements

- **1.2× speedup** over state-of-the-art baselines on extremely long sequences (>1M tokens)
- **26.4% memory reduction** compared to most memory-efficient baselines
- **Linear scaling** with the number of devices along the sequence dimension
- **Support for sequences up to 4M tokens** on 64×A800 GPUs


## 🛠️ Installation

### Prerequisites

- Docker
- CUDA-compatible GPUs
- InfiniBand network (for multi-node training)
- Shared storage accessible by all nodes

### Docker-based Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/BurstEngine.git
cd BurstEngine
git submodule update --init --recursive
```

2. **Set up environment variables:**
Create and configure `env.sh` with your network settings:
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export UCX_NET_DEVICES=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_5,mlx5_6"
```

**Important:** Replace `bond0` with your actual network interface name. Use `ifconfig` to check available interfaces:
```bash
ifconfig
```

**Example network configuration:**
```bash
# Node 0
bond0: flags=5187<UP,BROADCAST,RUNNING,MASTER,MULTICAST>  mtu 1500
      inet 10.0.2.12  netmask 255.255.255.0  broadcast 10.0.2.255

# Node 1  
bond0: flags=5187<UP,BROADCAST,RUNNING,MASTER,MULTICAST>  mtu 1500
      inet 10.0.2.13  netmask 255.255.255.0  broadcast 10.0.2.255
```

**Note:** Contact your hardware provider to ensure correct `NCCL_IB_HCA` values, as some InfiniBand HCAs are used for shared storage rather than program communication.

3. **Build the Docker image:**
```bash
docker build -t burst_engine:latest .
```

4. **Distribute the image to all nodes:**
On the build machine:
```bash
docker save burst_engine:latest > /shared/burst_engine.tar.gz
```

On each target node:
```bash
docker load -i /shared/burst_engine.tar.gz
```

## 🚀 Quick Start

### Check Node Availability

Before running experiments, check which GPU nodes are available:
```bash
# Modify check.sh to specify your node prefix and range
node_prefix="bjdx"  # Replace with your node prefix
# Check nodes bjdx1, bjdx2, bjdx3, bjdx4
bash check.sh
```

### Running End-to-End Training

1. **Configure your experiment:**
Edit `code/BurstEngine/apps/llama/multi_exp.sh`:
```bash
#!/bin/bash

sizes=( 1048576 2097152 )  # Sequence lengths to test
methods=(
  "burst"  # BurstEngine method
)
export NODES="bjdx1 bjdx2 bjdx3 bjdx4"  # Your node list
export MODEL="7b"  # Model size
export CP_SIZE=32  # Context parallel size
```

2. **Set environment variables:**
```bash
export PROJECT_DIR=/shared/sc_workspace/BE/
```

3. **Launch multi-node training:**
```bash
cd code/BurstEngine/apps/llama
bash multi_exp.sh
```

### Running Baselines

Compare BurstEngine with other methods:

```bash
# Megatron-LM
cd evaluation/baselines/Megatron-LM && bash multi_exp.sh

# Megatron-DeepSpeed  
cd evaluation/baselines/Megatron-DeepSpeed/script && bash multi_exp.sh

# Intern-Evo
cd evaluation/baselines/InternEvo && bash multi_exp.sh
```

### Running Attention Benchmarks

```bash
cd evaluation/kernel_bench/
bash bench_all.sh
```
If you want detail guidance for reproduction, please refer to [this](./HELP.md).
