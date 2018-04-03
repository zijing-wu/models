import os,re,shutil
from subprocess import PIPE, Popen
from multiprocessing import Pool, Queue, Process, Manager
from itertools import product
from time import sleep

PROCESS_NUMBER = 2
FILE_PER_BATCH = 10

train_images_path = os.path.join('.','data_retrieval','train')
test_images_path = os.path.join('.','data_retrieval','test')

des_train_features = os.path.join('.','train_features')
des_test_features = os.path.join('.','test_features')

gen_list = 'image_list'

'''for path in [des_train_features,des_test_features]:
    if os.path.exists(path):
       shutil.rmtree(path)
    os.makedirs(path)'''

def batch_gen_features(index,is_train):
    if is_train:
        gen_path = os.path.join(gen_list,"train_list"+str(index)+'.txt')
        des_path = des_train_features
    else:
        gen_path = os.path.join(gen_list,"test_list"+str(index)+'.txt')
        des_path = des_test_features

    print("Process image in %s"%(gen_path,))

    p = Popen('''
    python3 extract_features.py \
  --config_path delf_config_example.pbtxt \
  --list_images_path %s\
  --output_dir %s
    '''%(gen_path,des_path),shell=True, stdout=PIPE, stderr=PIPE)

    stdout, stderr = p.communicate()

    print(str(stdout))
    print(str(stderr))

train_images = os.listdir(train_images_path)
test_images = os.listdir(test_images_path)

train_total = int(len(train_images)/(FILE_PER_BATCH)) + 1
test_total = int(len(test_images)/FILE_PER_BATCH) + 1

with Pool(PROCESS_NUMBER) as p:
    p.starmap(batch_gen_features, product(range(train_total), [True]))
    p.close()
    p.join()

with Pool(PROCESS_NUMBER) as p:
    p.starmap(batch_gen_features, product(range(test_total), [False]))
    p.close()
    p.join()

print("Finish!!!!")
