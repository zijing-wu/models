#!/bin/bash

find data_retrieval/train/ -type f -name '*.jpg' > train_files.txt
find data_retrieval/test/ -type f -name '*.jpg' > test_files.txt
#ls -d -1 data_retrieval/train/*.* > index_files.txt
#ls -d -1 data_retrieval/test/*.* > test_files.txt
#ls -d -1 data_retrieval/test/*.* > all_files.txt
#ls -d -1 data_retrieval/train/*.* >> all_files.txt


wc -l train_files.txt
wc -l test_files.txt
#wc -l all_files.txt

echo "done."


