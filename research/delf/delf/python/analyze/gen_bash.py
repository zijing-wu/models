import os,sys
from subprocess import Popen

num_process = int(sys.argv[1])

test_image_path = os.path.join('..','examples','test_features_ds2')
test_file = os.listdir(test_image_path)

NUM_PER_PROCESS = int(len(test_file)/num_process) + 1

for i in range(num_process):
    command = "CUDA_VISIBLE_DEVICES=%d nohup python3 get_pair.py %d %d > gpu%d.log &"%(i,NUM_PER_PROCESS*i,NUM_PER_PROCESS*(i+1),i)
    Popen(command,shell=True)
