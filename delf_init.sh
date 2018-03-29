#!/bin/bash

cd research/slim/
sudo pip3 install -e .

cd ..
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`

# From tensorflow/models/research/delf/
cd delf/
${PATH_TO_PROTOC?}/bin/protoc delf/protos/*.proto --python_out=.

# From tensorflow/models/research/delf/
sudo pip3 install -e . 
# Install "delf" package.

# test
python3 -c 'import delf'

# From tensorflow/models/research/delf/delf/python/examples/
cd delf/python/examples/
mkdir parameters
cd parameters
wget http://download.tensorflow.org/models/delf_v1_20171026.tar.gz
tar -xvzf delf_v1_20171026.tar.gz

