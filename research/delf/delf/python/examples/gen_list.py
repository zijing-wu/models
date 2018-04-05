import os,csv
import argparse
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf

from tensorflow.python.platform import app
from delf import feature_io

count = 1
label2id = {}
with open('data_retrieval/train.csv') as file:
    csv_reader =  csv.reader(file)
    for row in csv_reader:
        count += 1
        if (row[2] not in label2id or label2id[row[2]][1] <= 600 ) and os.path.exists(os.path.join('train_features_ds2',row[0]+'.delf')):
            locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(os.path.join('train_features_ds2',row[0]+'.delf'))
            num_features_1 = locations_1.shape[0]
            if row[2] not in label2id or num_features_1 > label2id[row[2]][1]:
                label2id[row[2]] = (row[0],num_features_1)
        if count % 1000 == 0:
            print("finish %d"%(count,))

with open('./image_list.txt','w') as file:
    for k,v in label2id.items():
        file.write(v[0]+','+k+'\n')