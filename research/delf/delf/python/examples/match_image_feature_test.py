from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

import time,os,re

cmd_args = None

_DISTANCE_THRESHOLD = 0.8

train_feature_path = os.path.join('train_features')
test_feature_path = os.path.join('test_features')

line_pattern = re.compile('Found (\d{1,10}) inliers')
name_pattern = re.compile('(.*?)\.delf')

train_file,test_file = os.listdir(train_feature_path),os.listdir(test_feature_path)

if not os.path.exists("lines"):
    os.mkdir("lines")

des_file = os.path.join("lines")

def match(img1,img2):
    tf.logging.set_verbosity(tf.logging.INFO)

    start_time = time.time()

    # Read features.
    locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(
        img1)
    num_features_1 = locations_1.shape[0]
    tf.logging.info("Loaded image 1's %d features" % num_features_1)
    locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(
        img2)
    num_features_2 = locations_2.shape[0]
    tf.logging.info("Loaded image 2's %d features" % num_features_2)


    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)


    # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i, ]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i], ]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    print(locations_1_to_use.shape)
    print(locations_2_to_use.shape)

    return locations_1_to_use.shape[0],locations_2_to_use.shape[0]

argv = sys.argv[1:]
for i in range(int(argv[0]),int(argv[1])):
    if i >= len(test_file):
        break
    des_f = test_file[i][0:-5]
    if os.path.exists(os.path.join(des_file,des_f+'.txt')):
        continue
    f1 = os.path.join(test_feature_path, test_file[i])
    with open(os.path.join(des_file,des_f+'.txt'),'w') as file:
        for t_file in train_file:
            res = match(f1,os.path.join(train_feature_path, t_file))
            file.write(t_file[0:-5]+','+str(res)+'\n')