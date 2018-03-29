#!/bin/bash

PATH_TO_HOME=/models/
PATH_TO_PROTOC=${PATH_TO_HOME?}/lib
PATH_TO_RESEARCH=${PATH_TO_HOME?}/research
export PYTHONPATH=$PYTHONPATH:${PATH_TO_RESEARCH}
cd ${PATH_TO_RESEARCH?}/delf

${PATH_TO_PROTOC?}/bin/protoc delf/protos/*.proto --python_out=.

WORK_PATH=${PATH_TO_RESEARCH?}/delf/delf/python/examples
cd ${WORK_PATH} 

export PYTHONPATH=$PYTHONPATH:${PATH_TO_RESEARCH?}:${PATH_TO_RESEARCH?}/slim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/extras/CUPTI/lib64
export CUDA_HOME=/usr/local/cuda-9.0

