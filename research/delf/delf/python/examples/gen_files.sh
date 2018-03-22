#!/bin/bash


ls -lrt -d -1 $(find data_retrieval/index -type f ) > index_files.txt
ls -lrt -d -1 $(find data_retrieval/test -type f ) > test_files.txt
ls -lrt -d -1 $(find data_retrieval/test -type f ) > all_files.txt
ls -lrt -d -1 $(find data_retrieval/index -type f ) >> all_files.txt

wc -l index_files.txt
wc -l test_files.txt
wc -l all_files.txt

echo "done."


