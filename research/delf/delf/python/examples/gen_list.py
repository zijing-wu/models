
import os,csv

count = 1
label2id = {}
with open('data_retrieval/train.csv') as file:
    csv_reader =  csv.reader(file)
    for row in csv_reader:
         if row[2] not in label2id and os.path.exists(os.path.join('train_features_ds2',row[0]+'.delf')):
             label2id[row[2]] = row[0]

with open('./image_list.txt','w') as file:
    for k,v in label2id.items():
        file.write(v+','+k+'\n')