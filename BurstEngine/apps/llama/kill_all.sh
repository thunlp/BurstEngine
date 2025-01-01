#!/bin/bash

# 使用ps命令查找所有CMD为torchrun或python的进程，并使用awk提取进程ID
pids=$(ps -eo pid,cmd | awk '/torchrun|python/ && !/awk/ {print $1}')

# 循环遍历匹配到的进程ID，并逐个进行kill -9操作
for pid in $pids
do
    echo "Killing process $pid"
    kill -9 $pid
done
