#!/bin/bash


# train : 1218618
# 304654, 609309, 913963, 
# test : 115747 

#CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0 nohup python3 ibatch_gen_features.py 0 400 0 1000 data_retrieval/train train_features/ > gpu0.log &
#CUDA_VISIBLE_DEVICES=1
CUDA_VISIBLE_DEVICES=1 nohup python3 ibatch_gen_features.py 1 400 1000 2000 data_retrieval/train train_features/ > gpu1.log &
#CUDA_VISIBLE_DEVICES=2
CUDA_VISIBLE_DEVICES=2 nohup python3 ibatch_gen_features.py 2 400 2000 3000 data_retrieval/train train_features/ > gpu2.log &
#CUDA_VISIBLE_DEVICES=3
CUDA_VISIBLE_DEVICES=3 nohup python3 ibatch_gen_features.py 3 400 3000 4000 data_retrieval/train train_features/ > gpu3.log &

