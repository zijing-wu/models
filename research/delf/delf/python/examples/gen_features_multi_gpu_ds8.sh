#!/bin/bash


# train : 1218618
# 304654, 609309, 913963,
# 152327.25, 304654, 456981, 609309, 761636, 913963, 1066290, 1218618
# test : 115747


#CUDA_VISIBLE_DEVICES=0 nohup python3 ibatch_gen_features.py 0 400 0 1000 data_retrieval/train train_features/ > gpu0.log &
#CUDA_VISIBLE_DEVICES=1 nohup python3 ibatch_gen_features.py 1 400 1000 2000 data_retrieval/train train_features/ > gpu1.log &

CUDA_VISIBLE_DEVICES=0 nohup python3 ibatch_gen_features.py 01 400 0 76163 data_retrieval/train train_features_ds8/ > gpu01.log &
CUDA_VISIBLE_DEVICES=0 nohup python3 ibatch_gen_features.py 02 400 76163 152327 data_retrieval/train train_features_ds8/ > gpu02.log &

CUDA_VISIBLE_DEVICES=1 nohup python3 ibatch_gen_features.py 11 400 152327 228409 data_retrieval/train train_features_ds8/ > gpu11.log &
CUDA_VISIBLE_DEVICES=1 nohup python3 ibatch_gen_features.py 12 400 228409 304654 data_retrieval/train train_features_ds8/ > gpu12.log &

CUDA_VISIBLE_DEVICES=2 nohup python3 ibatch_gen_features.py 21 400 304654 380816 data_retrieval/train train_features_ds8/ > gpu21.log &
CUDA_VISIBLE_DEVICES=2 nohup python3 ibatch_gen_features.py 22 400 380816 456981 data_retrieval/train train_features_ds8/ > gpu22.log &

CUDA_VISIBLE_DEVICES=3 nohup python3 ibatch_gen_features.py 31 400 456981 533144 data_retrieval/train train_features_ds8/ > gpu31.log &
CUDA_VISIBLE_DEVICES=3 nohup python3 ibatch_gen_features.py 32 400 533144 609309 data_retrieval/train train_features_ds8/ > gpu32.log &

CUDA_VISIBLE_DEVICES=4 nohup python3 ibatch_gen_features.py 41 400 609309 685470 data_retrieval/train train_features_ds8/ > gpu41.log &
CUDA_VISIBLE_DEVICES=4 nohup python3 ibatch_gen_features.py 42 400 685470 761636 data_retrieval/train train_features_ds8/ > gpu42.log &

CUDA_VISIBLE_DEVICES=5 nohup python3 ibatch_gen_features.py 51 400 761636 837799 data_retrieval/train train_features_ds8/ > gpu51.log &
CUDA_VISIBLE_DEVICES=5 nohup python3 ibatch_gen_features.py 52 400 837799 913963 data_retrieval/train train_features_ds8/ > gpu52.log &

CUDA_VISIBLE_DEVICES=6 nohup python3 ibatch_gen_features.py 61 400 913963 1066290 data_retrieval/train train_features_ds8/ > gpu61.log &
CUDA_VISIBLE_DEVICES=6 nohup python3 ibatch_gen_features.py 62 400 1066290 1218618 data_retrieval/train train_features_ds8/ > gpu62.log &

#CUDA_VISIBLE_DEVICES=6 nohup python3 ibatch_gen_features.py 6 400 0 57873 data_retrieval/test test_features_ds8/ > gpu6.log &
CUDA_VISIBLE_DEVICES=7 nohup python3 ibatch_gen_features.py 71 400 0 57873 data_retrieval/test test_features_ds8/ > gpu71.log &
CUDA_VISIBLE_DEVICES=7 nohup python3 ibatch_gen_features.py 72 400 57873 115747 data_retrieval/test test_features_ds8/ > gpu72.log &

