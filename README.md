
# Execution Guideline


## Prepare the Docker image

1. Execute command to clone the github repo

```bash
git clone git@github.com:MayDomine/BE.git # or https://github.com/MayDomine/BE.git, it's based on your network environment
cd BE && git submodule update --init --recursive 
```

2. Set up the environment script 

```cat env.sh```

please replace `bond0` with your NIC name and specify the IB port via `NCCL_IB_HCA` environment variable.

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

`BE/check.sh` provide a script to check if a gpu node available

You can specify set node list number and prefix in this way

```bash
node_prefix="bjdx"
for i in  {1..2}; do
    node="${node_prefix}$i"
```
In this way, the script will check if bjdx1 and bjdx2 available.

You can modify `check.sh` and `./BurstEngine/apps/llama/multi_exp.sh` to ensure the settings are what you want.

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
