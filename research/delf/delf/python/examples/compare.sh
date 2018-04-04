#!/bin/bash

#python3 -m cProfile -o new.profile match_images_fast.py \
#  --features_1_path test_features/all_souls_000000.delf \
#  --features_2_path test_features/all_souls_000001.delf

# test: 115747 / 8 = 14468.375
# 0, 14468, 28936， 43405， 57873， 72341， 86810， 101278， 115747

#python3 -m cProfile -o new.profile match_images_all.py \
#  test_features/ test_features/ 0 20 10 lines_out/

CUDA_VISIBLE_DEVICES=0 nohup python3 match_images_all.py train_features/ test_features/ 0 14468 1000 lines_out/ > lgpu0.log &
CUDA_VISIBLE_DEVICES=1 nohup python3 match_images_all.py train_features/ test_features/ 14468 28936 1000 lines_out/ > lgpu1.log &
CUDA_VISIBLE_DEVICES=2 nohup python3 match_images_all.py train_features/ test_features/ 28936 43405 1000 lines_out/ > lgpu2.log &
CUDA_VISIBLE_DEVICES=3 nohup python3 match_images_all.py train_features/ test_features/ 43405 57873 1000 lines_out/ > lgpu3.log &
CUDA_VISIBLE_DEVICES=4 nohup python3 match_images_all.py train_features/ test_features/ 57873 72341 1000 lines_out/ > lgpu4.log &
CUDA_VISIBLE_DEVICES=5 nohup python3 match_images_all.py train_features/ test_features/ 72341 86810 1000 lines_out/ > lgpu5.log &
CUDA_VISIBLE_DEVICES=6 nohup python3 match_images_all.py train_features/ test_features/ 86810 101278 1000 lines_out/ > lgpu6.log &
CUDA_VISIBLE_DEVICES=7 nohup python3 match_images_all.py train_features/ test_features/ 101278 115747 1000 lines_out/ > lgpu7.log &





