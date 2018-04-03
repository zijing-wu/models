import os,re,shutil
from subprocess import PIPE, Popen
from multiprocessing import Pool, Queue, Process, Manager
from itertools import product
from time import sleep

PROCESS_NUMBER = 3

res_path = os.path.join('.','line_res')
if os.path.exists(res_path):
   shutil.rmtree(res_path)
os.makedirs(res_path)

train_image_path = os.path.join('..','examples','data_retrieval','train')
test_image_path = os.path.join('..','examples','data_retrieval','test')
train_feature_path = os.path.join('..','examples','train_features')
test_feature_path = os.path.join('..','examples','test_features')

train_size = len(os.listdir(train_image_path))

line_pattern = re.compile('Found (\d{1,10}) inliers')
name_pattern = re.compile('(.*?)\.jpg')

res_queue = Queue()

def excute(train_file,test_file):
    os.chdir('../examples')
    try:
        train = name_pattern.findall(train_file)[0]
        test = name_pattern.findall(test_file)[0]
    except:
        print("There are some errors with %s and %s"%(train_file,test_file))
        return


    train_image = os.path.join(train_image_path,train+'.jpg')
    test_image = os.path.join(test_image_path,test+'.jpg')
    train_feature = os.path.join(train_feature_path,train+'.delf')
    test_feature = os.path.join(test_feature_path,test+'.delf')

    p = Popen('''python match_images.py \
  --image_1_path %s \
  --image_2_path %s \
  --features_1_path %s \
  --features_2_path %s '''%(train_image,test_image,train_feature,test_feature),shell=True, stdout=PIPE, stderr=PIPE)

    stdout, stderr = p.communicate()

    print("%s and %s finish"%(train,test))

    res = line_pattern.findall(str(stderr))

    if len(res) == 0:
        res_queue.put([train,test,0])
    else:
        res_queue.put([train,test,res[0]])

def count():
    while finish_flag.value != 1 or not res_queue.empty():
        if res_queue.empty():
            print("Colleting process is sleepping")
            sleep(20)
        else:
            print("Collecting process is working")
            while not res_queue.empty():
                train_id,test_id,lines = res_queue.get()
                if test_id not in res_dict:
                    res_dict[test_id] = 0
                res_dict[test_id] += 1
                with open(os.path.join(res_path,test_id+'.txt'),'a') as file:
                    file.write(train_id+' '+str(lines)+'\n')

with Manager() as manager:
    res_dict = manager.dict()
    finish_flag = manager.Value('flag',0)
    count_p = Process(target=count)
    count_p.start()
    with Pool(PROCESS_NUMBER) as p:
        p.starmap(excute,product(os.listdir(train_image_path),os.listdir(test_image_path)))
        p.close()
        p.join()
        finish_flag.value = 1
    count_p.join()

    with open('log.txt','w') as file:
        for k,v in res_dict.items():
            if v!= train_size:
                file.write("File %s is not totally finished\n"%(k,))
