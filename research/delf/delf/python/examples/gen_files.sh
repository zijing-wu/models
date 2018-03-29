#!/bin/bash

ls -d -1 data_retrieval/index/*.* > index_files.txt
ls -d -1 data_retrieval/test/*.* > test_files.txt
ls -d -1 data_retrieval/test/*.* > all_files.txt
ls -d -1 data_retrieval/index/*.* >> all_files.txt


wc -l index_files.txt
wc -l test_files.txt
wc -l all_files.txt

echo "done."


