#!/bin/bash
nsys profile --capture-range=cudaProfilerApi -t cuda,nvtx,cublas  -o $1 bash $2
