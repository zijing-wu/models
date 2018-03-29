#!/bin/bash


python3 extract_features.py --config_path delf_config_example.pbtxt --list_images_path index_files.txt --output_dir index_features

python3 extract_features.py --config_path delf_config_example.pbtxt --list_images_path test_files.txt --output_dir test_features


