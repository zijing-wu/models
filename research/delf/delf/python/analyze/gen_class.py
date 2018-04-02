import csv,os,json

path_to_cv = os.path.join('..','examples','data','cv_images')
path_to_train = os.path.join(path_to_cv,"train")
path_to_test = os.path.join(path_to_cv,"test")

train_files = set(os.listdir(path_to_train))
test_files = set(os.listdir(path_to_test))

train_dict = {}
test_dict = {}

train_file_count = len(train_files)
test_file_count = len(test_files)

with open('../train.csv') as csvfile:
    train_reader = csv.reader(csvfile)
    count = 0
    for row in train_reader:
        filename = row[0] + '.jpg'
        if filename in train_files:
            train_file_count -= 1
            train_dict[filename] = row[2]
        elif filename in test_files:
            test_file_count -= 1
            test_dict[filename] = row[2]

if train_file_count != 0:
    print("There are %d train files not classified",(train_file_count,))
if test_file_count != 0:
    print("There are %d test files not classified",(test_file_count,))

with open('./train_dict.json','w') as file:
    file.write(json.dumps(train_dict))

with open('./test_dict.json','w') as file:
    file.write(json.dumps(test_dict))
