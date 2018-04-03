#!/bin/bash

nohup CUDA_VISIBLE_DEVICES=0 python3 ibatch_gen_features.py 0 10 0 15 data_retrieval/train train_features/ > gpu0.log &
nohup CUDA_VISIBLE_DEVICES=1 python3 ibatch_gen_features.py 0 10 0 15 data_retrieval/train train_features/ > gpu1.log &
nohup CUDA_VISIBLE_DEVICES=2 python3 ibatch_gen_features.py 0 10 0 15 data_retrieval/train train_features/ > gpu2.log
nohup CUDA_VISIBLE_DEVICES=3 python3 ibatch_gen_features.py 0 10 0 15 data_retrieval/train train_features/ > gpu3.log
