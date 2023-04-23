#! /bin/bash

# run this DDP code with multiGPUs on a single Node
# 1. request resources from interactive mode or OOD
 
conda activate pytorch/1.10
python multigpu.py 50 10
