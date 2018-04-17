
import os,csv

MIN_COUNT = 20
FEATURE_DIR = 'train_features_ds8'
OUT_FILE = 'image_list_test_ds8.txt'
MAX_CLASS = 20
GEN_PATH = True

count = 1
label2id = {}
with open('data_retrieval/train.csv') as file:
    csv_reader =  csv.reader(file)
    for row in csv_reader:
         if os.path.exists(os.path.join(FEATURE_DIR, row[0]+'.delf')):
             if row[2] not in label2id:
                 label2id[row[2]] = []
             label2id[row[2]].append([row[0]])

print("read train.csv done.")

skip_num, gen_num, class_num = 0, 0, 0
with open(OUT_FILE, 'w') as file:
    for k,v in label2id.items():
        if len(v) >= MIN_COUNT:
            class_num += 1
            if(class_num > MAX_CLASS):
                break
            for cur_v in v:
                gen_num += 1
                if(GEN_PATH):
                    file.write(FEATURE_DIR+"/"+cur_v[0]+'.delf'+'\n')   
                else:
                    file.write(cur_v[0]+','+k+'\n')
        else:
            skip_num += 1
print("gen files: %d; skiped: %d; class num:%d"% (gen_num, skip_num, class_num))
