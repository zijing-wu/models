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

def extract_features(dir_name, files, i_start, i_end):
    dict_features_index = {}
    for i in range(i_start, i_end):
        if(i % 100 == 0):
            tf.logging.info("loading features...(%d/%d)"%(i,i_end-i_start))
        cur_features_file = dir_name + '/' + files[i]
        basename = os.path.basename(cur_features_file)
        basename = os.path.splitext(basename)[0]
        try:
            locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(
                cur_features_file)
        except:
            print("load feature get error, skip...[%s]"%(cur_features_file))
            continue
        dict_features_index[basename] = {"loc": locations_1, "des": descriptors_1}
    tf.logging.info("loading features done. size:%d" % (len(dict_features_index)))
    return dict_features_index

_INLIERS_THRESHOLD = 150
def main():
    #if len(sys.argv) != 7:
    #    print('Syntax: {} <train_dir/> <test_dir/> <test_start> <test_end> <batch_size> <out_dir/>'.format(sys.argv[0]))
    #    sys.exit(0)
    #(train_dir, test_dir, test_start, test_end, batch_size, out_dir) = sys.argv[1:]
    (train_dir, test_dir, test_start, test_end, batch_size, out_dir) = ('ox_train_features/','ox_train_features/',0,20,20,'lines_out')
    test_start=int(test_start)
    test_end=int(test_end)
    batch_size=int(batch_size)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tf.logging.set_verbosity(tf.logging.INFO)

    train_dir_name = os.path.abspath(train_dir)
    train_files = [f for f in os.listdir(train_dir_name) if isfile(join(train_dir_name, f))]
    test_dir_name = os.path.abspath(test_dir)
    test_files = [f for f in os.listdir(test_dir_name) if isfile(join(test_dir_name, f))]

    for s in range(test_start, test_end, batch_size):
        s_end = min(min(test_end, s + batch_size),len(test_files))
        tf.logging.info("====processing test batch...(%d-%d)/(%d-%d)====" % (s, s_end, test_start, test_end))
        dict_features_test = extract_features(test_dir_name, test_files, s, s_end)

        n=0
        N=len(dict_features_test)
        for test_id in dict_features_test:  # 1
            n+=1
            tf.logging.info("==processing test in batch...[%s](%d/%d)" % (test_id,n,N))
            cur_out = []
            test_feature = dict_features_test[test_id]
            locations_1 = test_feature['loc']
            descriptors_1 = test_feature['des']

            d1_tree = cKDTree(descriptors_1)
            #print("shape:%d,%d\n"%(descriptors_1.shape[0],descriptors_1.shape[1]))
            num_features_1 = locations_1.shape[0]

            for t in range(0, len(train_files), batch_size):
                t_end = min(t + batch_size,len(train_files))
                tf.logging.info("   train processing...(%d-%d)/%d" % (t, t_end, len(train_files)))
                dict_features_train = extract_features(train_dir_name, train_files, t, t_end)

                for train_id in dict_features_train:  # 2
                    train_feature = dict_features_train[train_id]
                    locations_2 = train_feature['loc']
                    descriptors_2 = train_feature['des']
                    num_features_2 = locations_2.shape[0]
                    _, indices = d1_tree.query(
                        descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

                    #locations_2_to_use = np.array([
                    #                                  locations_2[i,]
                    #                                  for i in range(num_features_2)
                    #                                  if indices[i] != num_features_1
                    #                                  ])
                    locations_1_to_use = np.array([
                                                      locations_1[indices[i],]
                                                      for i in range(num_features_2)
                                                      if indices[i] != num_features_1
                                                      ])
                    '''
                    try:
                        _, inliers = ransac(
                            (locations_1_to_use, locations_2_to_use),
                            AffineTransform,
                            min_samples=3,
                            residual_threshold=20,
                            max_trials=1000)
                    except:
                        tf.logging.info("error, skip...[%s-%s]" % (train_id, test_id))
                        continue
                    if (inliers is None or train_id == test_id):
                        tf.logging.info("inliners is none, skip...[%s-%s]" % (train_id, test_id))
                        continue
                    inliers_sum = sum(inliers)
                    tf.logging.info('%s-%s: found %d inliers; shape %d, %d' % (train_id, test_id, inliers_sum, locations_1_to_use.shape[0],locations_2_to_use.shape[0]))
                    if (inliers_sum > _INLIERS_THRESHOLD):
                        cur_out.append((train_id, inliers_sum))
                    '''
                    inliers_sum = locations_1_to_use.shape[0]
                    if (inliers_sum > _INLIERS_THRESHOLD):
                        tf.logging.info("[%s-%s]: shape: %d" % (train_id, test_id, inliers_sum))
                        cur_out.append((train_id, inliers_sum))


            out_file_path = out_dir + "/" + test_id + ".txt"
            with open(out_file_path, 'w') as the_file:
                for (train_id, inliers_sum) in cur_out:
                    the_file.write("%s,%s\n"% (train_id, inliers_sum))

if __name__ == '__main__':
    main()
