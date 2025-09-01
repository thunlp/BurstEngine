FROM nvcr.io/nvidia/pytorch:24.07-py3

USER root

RUN apt-get update && apt-get install -y --no-install-recommends pdsh

COPY ./BMTrain /BMTrain 

COPY ./BurstEngine /BurstEngine
RUN bash /BurstEngine/apps/llama/pre.sh

RUN pip install /BMTrain -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

ENV FLASH_ATTENTION_FORCE_BUILD TRUE

COPY ./flash-attention /flash-attention
RUN pip install /flash-attention -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY ./ml-cross-entropy /ml-cross-entropy
RUN pip install /ml-cross-entropy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple


COPY ./Burst-Attention /Burst-Attention
RUN pip install /Burst-Attention -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install triton==3.0.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY ./llama-7b /llama-7b
COPY ./evaluation/baselines/InternEvo /InternEvo
COPY ./evaluation/baselines/Megatron-DeepSpeed /Megatron-DeepSpeed
COPY ./evaluation/baselines/Megatron-LM /Megatron-LM
COPY ./evaluation/kernel_bench/ /kernel_bench
COPY ./env.sh /env.sh
COPY ./data /data
