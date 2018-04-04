#!/bin/bash

#python3 -m cProfile -o new.profile match_images_fast.py \
#  --features_1_path test_features/all_souls_000000.delf \
#  --features_2_path test_features/all_souls_000001.delf

python3 -m cProfile -o new.profile match_images_all.py \
  test_features/ test_features/ 0 20 10 lines_out/



