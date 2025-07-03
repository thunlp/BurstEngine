FROM nvcr.io/nvidia/pytorch:24.07-py3
COPY ./BMTrain /BMTrain
RUN pip install /BMTrain

ENV FLASH_ATTENTION_FORCE_BUILD TRUE

COPY ./flash-attention /flash-attention
RUN pip install /flash-attention

COPY ./ml-cross-entropy /ml-cross-entropy
RUN pip install /ml-cross-entropy

COPY ./BurstEngine /BurstEngine
RUN bash /BurstEngine/apps/llama/pre.sh

COPY ./Burst-Attention /Burst-Attention
RUN pip install /Burst-Attention
RUN pip install triton==3.0.0

COPY ./llama-7b /llama-7b
COPY ./evaluation/baselines/InternEvo /InternEvo
COPY ./evaluation/baselines/Megatron-DeepSpeed /Megatron-DeepSpeed
COPY ./evaluation/baselines/Megatron-LM /Megatron-LM
COPY ./evaluation/kernel_bench/ /kernel_bench
