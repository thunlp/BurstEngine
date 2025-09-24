
# Execution Guideline


## Prepare the Docker image

1. Execute command to clone the github repo

```bash
git clone git@github.com:MayDomine/BE.git # or https://github.com/MayDomine/BE.git, it's based on your network environment
cd BE && git submodule update --init --recursive 
```

2. Set up the environment script 

All related environment variables are stored in BE/env.sh, and the training script will automatically source this file upon launch to ensure they take effect.
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 # you do not need to modify this.
export UCX_NET_DEVICES=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_5,mlx5_6"
```

Above is an example for env.sh. You must replace all instances of "bond0" with your own `IFNAME`. You can use the `ifconfig` command to see all network interface names and select the one you use for torchrun's TCP handshake.

```shell
(base) root@node0:/working_dir# ifconfig
bond0: flags=5187<UP,BROADCAST,RUNNING,MASTER,MULTICAST>  mtu 1500
        inet 10.0.2.12  netmask 255.255.255.0  broadcast 10.0.2.255

# Node1  
(base) root@node1:/working_dir# ifconfig
bond0: flags=5187<UP,BROADCAST,RUNNING,MASTER,MULTICAST>  mtu 1500
        inet 10.0.2.13  netmask 255.255.255.0  broadcast 10.0.2.255
```
In this case, it means the node0 and node1 can communicate with each other by using `bond0` as the network interface. 


About `NCCL_IB_HCA`: You need to contact your hardware provider to ensure you set the correct value, as some InfiniBand HCAs are used for shared storage filesystems rather than program communication.

3. Build the image

```bash
docker build -t burst_engine:latest .
```

4. Distribute the image

Execute the command below on the machine where you built the image
```bash 
# /shared should be some shared storage/directory that can be assessed by all machines.
docker save burst_engine:latest > /shared/burst_engine.tar.gz
```
Execute the command below on the machine where you want distribute the image

```bash
docker load -i /shared/burst_engine.tar.gz
```


## Running End-to-End Training experiment

### BurstEngine

1. Specify nodes that you want to run.

    - check.sh provide a script to check if a gpu node available. 
   
    - You need to specify node number and prefix in check.sh.

    - You can check hostname by using `cat /etc/hosts`, A output may like
```
bjdx1 10.0.2.1
bjdx2 10.0.2.2
bjdx3 10.0.2.3
bjdx4 10.0.2.4
```
In this case `bjdx` is the node_prefix, and number list should be {1..4}

```bash
node_prefix="bjdx" # check.sh:7
...
for i in  {1..4}; do #check.sh:71
    node="${node_prefix}$i"
```
Then you can modify the check.sh and ensure that the script will check if `bjdx1,bjdx2,bjdx3,bjdx4` available.

About settings related with training experiment, you need to modify `./BurstEngine/apps/llama/multi_exp.sh` to ensure the settings are what you want.

```bash
#!/bin/bash

sizes=( 1048576 2097152 )
methods=(
  "burst"
)
ablation=(
  "false"
)
export PROFILE="false"
export NODES="bjdx1 bjdx2 bjdx3 bjdx4"
export MODEL="7b"
DOCKER_DIR=/BurstEngine/apps/llama
export LOG_FILE=$DOCKER_DIR/sl_ckpt_exp.log
echo $LOG_FILE
for method in ${methods[@]}; do
  for ablation in ${ablation[@]}; do
  echo "Running method $method" >> summary.txt
  for size in ${sizes[@]}; do
    for cp in 32; do
      export WORLD_SIZE=$((cp > 8 ? cp / 8 : 1))
      export CP_SIZE=$cp
      export ABLATION=$ablation
      echo "Running size $size with method $method" >> summary.txt
      bash ./submit.sh "bash build_run.sh $size $method " 
      done
    done
  done
done
```

 In this case, the script will launch experiments using four nodes (32GPUs) and evaluate under 1M and 2M sequence lengths.
2. Launch the End-to-End Training script.

Set the env by using command below:

```bash
export PROJECT_DIR=/shared/sc_workspace/BE/
```

Execute the script to launch multi node experiment.
```bash
cd BurstEngine/apps/llama && bash ./multi_exp.sh
```

### Baselines
1. Megatron-LM

     Entrypoint: `cd evaluation/baselines/Megatron-LM && bash multi_exp.sh`

2. Megatron-DeepSpeed

     Entrypoint: `cd evaluation/baselines/Megatron-DeepSpeed/script && bash mutli_exp.sh`

3. Intern-Evo

     Entrypoint: `cd evaluation/baselines/InternEvo bash multi_exp.sh`

All entrypoint should be execute in the bare-metal machine instead of docker container since the entrypoint will use pdsh to launch docker image and execute the training script.


## Running the benchmark of Distribtued Attention

Entrypoint: `cd evaluation/kernel_bench/ && bash bench_all.sh`
