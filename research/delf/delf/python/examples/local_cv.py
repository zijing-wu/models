import sys,os,json,csv

def main():

    if len(sys.argv) >= 2:
        train_path = sys.argv[1]
    else:
        train_path = os.path.join('.','data_retrieval','train.csv')

    if len(sys.argv) == 3:
        res_path = sys.argv[2]
    else:
        res_path = os.path.join('.','data_retrieval','sample_submission.csv')
        print("Load default submission csv %s"%(res_path,))

    id2label_dict_path = os.path.join('.','id2label.json')
    if os.path.exists(id2label_dict_path):
        with open(id2label_dict_path,'r') as file:
            id2label_dict = json.loads(file.read())
    else:
        id2label_dict = {}
        with open(train_path,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == 'id':
                    continue
                id2label_dict[row[0]]= row[2]
        with open(id2label_dict_path,'w') as file:
            file.write(json.dumps(id2label_dict))

    with open(res_path) as file:
        csv_reader = csv.reader(file)
        N = 0
        GAP = 0
        for row in csv_reader:
            if row[0] == 'id':
                continue
            N += 1
            id = row[0]
            label = row[1].split()[0]
            rel = row[1].split()[1]
            if label == id2label_dict[id]:
                GAP += float(rel)

    print(GAP/N)



if __name__ == '__main__':
    main()
