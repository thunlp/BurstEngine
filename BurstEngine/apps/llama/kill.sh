#!/bin/bash

# 使用ps命令查找所有CMD为torchrun或python的进程，并使用awk提取进程ID
ps -aux | awk '/pretrain_llama_hug_tokenizer.py/ {print $2}' | xargs kill -9
