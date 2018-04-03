#!/bin/bash


# train : 1218618
# 304654, 609309, 913963,
# 152327.25, 304654, 456981, 609309, 761636, 913963, 1066290, 1218618
# test : 115747


#CUDA_VISIBLE_DEVICES=0 nohup python3 ibatch_gen_features.py 0 400 0 1000 data_retrieval/train train_features/ > gpu0.log &
#CUDA_VISIBLE_DEVICES=1 nohup python3 ibatch_gen_features.py 1 400 1000 2000 data_retrieval/train train_features/ > gpu1.log &

CUDA_VISIBLE_DEVICES=0 nohup python3 ibatch_gen_features.py 0 400 0 152327 data_retrieval/train train_features/ > gpu0.log &
CUDA_VISIBLE_DEVICES=1 nohup python3 ibatch_gen_features.py 1 400 152327 304654 data_retrieval/train train_features/ > gpu1.log &
CUDA_VISIBLE_DEVICES=2 nohup python3 ibatch_gen_features.py 2 400 304654 456981 data_retrieval/train train_features/ > gpu2.log &
CUDA_VISIBLE_DEVICES=3 nohup python3 ibatch_gen_features.py 3 400 456981 609309 data_retrieval/train train_features/ > gpu3.log &
CUDA_VISIBLE_DEVICES=4 nohup python3 ibatch_gen_features.py 4 400 609309 761636 data_retrieval/train train_features/ > gpu4.log &
CUDA_VISIBLE_DEVICES=5 nohup python3 ibatch_gen_features.py 5 400 761636 913963 data_retrieval/train train_features/ > gpu5.log &
#CUDA_VISIBLE_DEVICES=6 nohup python3 ibatch_gen_features.py 6 400 913963 1066290 data_retrieval/train train_features/ > gpu6.log &
#CUDA_VISIBLE_DEVICES=7 nohup python3 ibatch_gen_features.py 7 400 1066290 1218618 data_retrieval/train train_features/ > gpu7.log &

CUDA_VISIBLE_DEVICES=6 nohup python3 ibatch_gen_features.py 6 400 0 57873 data_retrieval/test test_features/ > gpu6.log &
CUDA_VISIBLE_DEVICES=7 nohup python3 ibatch_gen_features.py 7 400 57873 115747 data_retrieval/test test_features/ > gpu7.log &

