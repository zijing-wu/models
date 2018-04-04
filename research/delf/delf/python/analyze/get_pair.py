import os,re,shutil,sys
from subprocess import PIPE, Popen
from multiprocessing import Pool, Queue, Process, Manager
from itertools import product
from time import sleep

train_image_path = os.path.join('..','examples','data_retrieval','train')
test_image_path = os.path.join('..','examples','data_retrieval','test')
train_feature_path = os.path.join('..','examples','train_features')
test_feature_path = os.path.join('..','examples','test_features')

line_pattern = re.compile('Found (\d{1,10}) inliers')
name_pattern = re.compile('(.*?)\.jpg')

train_file,test_file = os.listdir(train_image_path),os.listdir(test_image_path)

if not os.path.exists("lines"):
    os.mkdir("lines")

des_file = os.path.join("lines")

def excute(test_file,train_file):
    os.chdir('../examples')
    try:
        train = name_pattern.findall(train_file)[0]
        test = name_pattern.findall(test_file)[0]
    except:
        print("There are some errors with %s and %s"%(train_file,test_file))
        return

    train_feature = os.path.join(train_feature_path,train+'.delf')
    test_feature = os.path.join(test_feature_path,test+'.delf')

    p = Popen('''python match_images.py \
  --features_1_path %s \
  --features_2_path %s '''%(train_feature,test_feature),shell=True, stdout=PIPE, stderr=PIPE)

    stdout, stderr = p.communicate()

    print("%s and %s finish"%(train,test))
    print(str(stderr))

    res = line_pattern.findall(str(stderr))

    os.chdir('../analyze')
    if len(res) == 1 :
        return res[0]
    else:
        return 0


argv = sys.argv[1:]
for i in range(int(argv[0]),int(argv[1])):
    if i >= len(test_file):
        break
    des_f = test_file[i][0:-4]
    print(des_f)
    if os.path.exists(os.path.join(des_file,des_f+'.txt')):
        continue
    with open(os.path.join(des_file,des_f+'.txt'),'w') as file:
        for t_file in train_file:
            res = excute(test_file[i],t_file)
            if res >= 16:
                file.write(t_file[0:-4]+','+str(res)+'\n')
