# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Matches two images using their DELF features.

The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.

The DELF features can be extracted using the extract_features.py script.
"""

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
from os.path import isfile, join

import glob, os

cmd_args = None

_DISTANCE_THRESHOLD = 0.8

PLOT_FIG = False
#INDEX_FEATURE_FOLDER = 'data/oxford5k_features'
#QUERY_FEATURE_FOLDER = 'data/oxford5k_features'
#OUT_PUT_FILE = 'out.csv'

def extract_features(dir_name):
    #files = os.listdir(dir_name)
    #os.chdir(dir_name)
    #files = glob.glob("*.delf")
    dir_name = os.path.abspath(dir_name)
    files = [f for f in os.listdir(dir_name) if isfile(join(dir_name, f))]
    dict_features_index = {}
    for i in range(len(files)):
        if(i>1000): break

        if(i % 100 == 0):
            tf.logging.info("loading features...(%d/%d)"%(i,len(files)))
        cur_features_file = dir_name + '/' + files[i]
        basename = os.path.basename(cur_features_file)
        basename = os.path.splitext(basename)[0]
        #basename = os.path.splitext(cur_features_file)[0]
        try:
            locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(
                cur_features_file)
        except:
            print("get error, skip...[%s]"%(cur_features_file))
            continue
        dict_features_index[basename] = {"loc": locations_1, "des": descriptors_1}
    tf.logging.info("loading features done.")
    return dict_features_index

_INLIERS_THRESHOLD = 30
def main():
    if len(sys.argv) != 4:
        print('Syntax: {} <index_dir/> <test_dir/> <out.csv>'.format(sys.argv[0]))
        sys.exit(0)
    (INDEX_FEATURE_FOLDER, QUERY_FEATURE_FOLDER, OUT_PUT_FILE) = sys.argv[1:]

    tf.logging.set_verbosity(tf.logging.INFO)

    dict_features_index = extract_features(INDEX_FEATURE_FOLDER)
    dict_features_query = extract_features(QUERY_FEATURE_FOLDER)

    output={}
    for query_id in dict_features_query: #1
        output[query_id] = []
        query_feature = dict_features_query[query_id]
        locations_1 = query_feature['loc']
        descriptors_1 = query_feature['des']
        d1_tree = cKDTree(descriptors_1)
        num_features_1 = locations_1.shape[0]
        for index_id in dict_features_index: #2
            index_feature = dict_features_index[index_id]
            locations_2 = index_feature['loc']
            descriptors_2 = index_feature['des']
            num_features_2 = locations_2.shape[0]
            _, indices = d1_tree.query(
                descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

            locations_2_to_use = np.array([
                                          locations_2[i,]
                                          for i in range(num_features_2)
                                          if indices[i] != num_features_1
                                          ])
            locations_1_to_use = np.array([
                                          locations_1[indices[i],]
                                          for i in range(num_features_2)
                                          if indices[i] != num_features_1
                                          ])
            try:
                _, inliers = ransac(
                (locations_1_to_use, locations_2_to_use),
                AffineTransform,
                min_samples=3,
                residual_threshold=20,
                max_trials=1000)
            except:
                print("error, skip...[%s-%s]"%(query_id, index_id))
                continue
            if(inliers is None or query_id==index_id):
                print("inliners is none, skip...[%s-%s]"%(query_id, index_id))
                continue
            inliers_sum = sum(inliers)
            tf.logging.info('%s-%s: found %d inliers' % (query_id, index_id, inliers_sum))
            if(inliers_sum > _INLIERS_THRESHOLD):
                output[query_id].append(index_id)


    with open(OUT_PUT_FILE, 'w') as the_file:
        for query_id in output:
            index_list = output[query_id]
            row = ' '.join(index_list)
            the_file.write("%s,%s\n"% (query_id, row))

if __name__ == '__main__':
    main()
