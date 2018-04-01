import os,re,shutil
from subprocess import PIPE, Popen

FILE_PER_BATCH = 400

train_images_path = os.path.join('.','data_retrieval','train')
test_images_path = os.path.join('.','data_retrieval','test')

des_train_features = os.path.join('.','train_features')
des_test_features = os.path.join('.','test_features')

gen_list = 'batch_list_images.txt'

for path in [des_train_features,des_test_features]:
    if os.path.exists(path):
       shutil.rmtree(path)
    os.makedirs(path)

def batch_gen_features(image_list,src_path,des_path):
    with open(gen_list,'w') as file:
        for image in image_list:
            file.write(os.path.join(src_path, image)+'\n')
    p = Popen('''
    python extract_features.py \
  --config_path delf_config_example.pbtxt \
  --list_images_path %s\
  --output_dir %s
    '''%(gen_list,des_path),shell=True, stdout=PIPE, stderr=PIPE)

    stdout, stderr = p.communicate()

    print(stdout)
    print(stderr)

train_images = os.listdir(train_images_path)
test_images = os.listdir(test_images_path)

for s in range(0,len(train_images),FILE_PER_BATCH):
    batch_gen_features(train_images[s:s+FILE_PER_BATCH],train_images_path,des_train_features)

for s in range(0,len(test_images),FILE_PER_BATCH):
    batch_gen_features(test_images[s:s+FILE_PER_BATCH],test_images_path,des_test_features)

