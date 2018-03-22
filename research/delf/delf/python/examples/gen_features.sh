#!/bin/bash


export PYTHONPATH=$PYTHONPATH:/Users/haoyuhe/github/google_landmark/models/research:/Users/haoyuhe/github/google_landmark/models/research/slim

python extract_features.py --list_images_path all_files.txt --output_dir all_features

