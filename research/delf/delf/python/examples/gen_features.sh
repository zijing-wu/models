#!/bin/bash


nohup python3 extract_features.py --config_path delf_config_example.pbtxt --list_images_path train_files.txt --output_dir train_features > train_nohup.log &


nohup python3 extract_features.py --config_path delf_config_example.pbtxt --list_images_path test_files.txt --output_dir test_features > test_nohup.log &


