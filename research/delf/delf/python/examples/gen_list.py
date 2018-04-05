
import os,csv

count = 1
label2id = {}
with open('data_retrieval/train.csv') as file:
    csv_reader =  csv.reader(file)
    for row in csv_reader:
         if os.path.exists(os.path.join('train_features_ds2',row[0]+'.delf')):
             if row[2] not in label2id:
                 label2id[row[2]] = []
             label2id[row[2]].append([row[0]])

with open('./image_list.txt','w') as file:
    for k,v in label2id.items():
        if len(v) > 10:
            file.write(v[0]+','+k+'\n')