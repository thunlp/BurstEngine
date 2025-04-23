FROM nvcr.io/nvidia/pytorch:24.07-py3
COPY ./BMTrain /BMTrain
RUN pip install /BMTrain

COPY ./flash-attention /flash-attention
RUN pip install /flash-attention

COPY ./ml-cross-entropy /ml-cross-entropy
RUN pip install /ml-cross-entropy

COPY ./BurstEngine /BurstEngine
RUN bash ./BurstEngine/apps/llama/pre.sh

