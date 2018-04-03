import sys,os,re,shutil
from subprocess import PIPE, Popen

def batch_gen_features(image_list,src_path,des_path,id):
    gen_list = str(id)+'_batch_list_images.txt'
    with open(gen_list,'w') as file:
        for image in image_list:
            file.write(os.path.join(src_path, image)+'\n')
        p = Popen('''
            python3 extract_features.py \
            --config_path delf_config_example.pbtxt \
            --list_images_path %s\
            --output_dir %s
            '''%(gen_list,des_path),shell=True, stdout=PIPE, stderr=PIPE)
        
        stdout, stderr = p.communicate()
        
        print(str(stdout))
    print(str(stderr))

def main():
    if len(sys.argv) != 7:
        print('Syntax: {} <ID> <FILE_PER_BATCH> <I_START> <I_END> <IMAGE_DIR/> <OUT_DIR/>'.format(sys.argv[0]))
        sys.exit(0)
    (ID, FILE_PER_BATCH, I_START, I_END, IMAGE_DIR, OUT_DIR) = sys.argv[1:]
    FILE_PER_BATCH = int(FILE_PER_BATCH)
    I_START = int(I_START)
    I_END = int(I_END)
    

    images_path = IMAGE_DIR
    des_features = OUT_DIR

    images = os.listdir(images_path)

    for s in range(I_START,I_END,FILE_PER_BATCH):
        s_end = min(I_END, s+FILE_PER_BATCH)
        batch_gen_features(images[s:s_end],images_path,des_features,ID)

if __name__ == '__main__':
    main()






