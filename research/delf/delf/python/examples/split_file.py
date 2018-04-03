import os

FILE_PER_BATCH = 10

train_images_path = os.path.join('.','data_retrieval','train')
test_images_path = os.path.join('.','data_retrieval','test')

list_path = 'image_list'

train_images = os.listdir(train_images_path)
test_images = os.listdir(test_images_path)

if not os.path.exists(list_path):
    os.makedirs(list_path)

total_list = int(len(train_images)/(FILE_PER_BATCH)) + 1
for i in range(total_list):
    file_to_write = os.path.join(list_path,"train_list"+str(i)+'.txt')
    with open(file_to_write,'w') as file:
        for image in train_images[i*FILE_PER_BATCH:(i+1)*FILE_PER_BATCH]:
            file.write(os.path.join(train_images_path, image)+'\n')

test_total_list = int(len(test_images)/FILE_PER_BATCH) + 1
for i in range(test_total_list):
    file_to_write = os.path.join(list_path,"test_list"+str(i)+'.txt')
    with open(file_to_write,'w') as file:
        for image in test_images[i*FILE_PER_BATCH:(i+1)*FILE_PER_BATCH]:
            file.write(os.path.join(test_images_path,image)+'\n')
