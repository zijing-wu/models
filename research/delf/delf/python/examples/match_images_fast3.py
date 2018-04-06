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

import sys
import numpy as np
from scipy.spatial import cKDTree
import tensorflow as tf

from delf import feature_io
from os.path import isfile, join
import datetime
import time

import os

#cmd_args = None

_DISTANCE_THRESHOLD = 0.8

PLOT_FIG = False

def extract_features(dir_name, files, i_start, i_end):
    dict_features_index = {}
    for i in range(i_start, i_end):
        if(i % 100 == 0):
            print("loading features...(%d/%d)"%(i,i_end-i_start))
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
    print("loading features done. size:%d" % (len(dict_features_index)))
    return dict_features_index

_INLIERS_THRESHOLD = 150
def main():
    if len(sys.argv) != 8:
        print('Syntax: {} <train_dir/> <test_dir/> <test_start> <test_end> <batch_size> <out_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (train_dir, test_dir, test_start, test_end, test_batch_size, train_batch_size, out_dir) = sys.argv[1:]
    #(train_dir, test_dir, test_start, test_end, test_batch_size, train_batch_size, out_dir) = ('ox_train_features/','ox_train_features/',0,100,20,100,'lines_out')
    test_start=int(test_start)
    test_end=int(test_end)
    test_batch_size=int(test_batch_size)
    train_batch_size = int(train_batch_size)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tf.logging.set_verbosity(tf.logging.INFO)

    train_dir_name = os.path.abspath(train_dir)
    train_files = [f for f in os.listdir(train_dir_name) if isfile(join(train_dir_name, f))]
    test_dir_name = os.path.abspath(test_dir)
    test_files = [f for f in os.listdir(test_dir_name) if isfile(join(test_dir_name, f))]

    test_N = test_end-test_start
    train_N = len(train_files)
    total_N = test_N*train_N
    t0 = time.time()

    for s in range(test_start, test_end, test_batch_size):
        s_end = min(min(test_end, s + test_batch_size),len(test_files))
        print("====processing test batch...(%d-%d)/(%d-%d)====" % (s, s_end, test_start, test_end))
        dict_features_test = extract_features(test_dir_name, test_files, s, s_end)

        cur_out = {}

        cur_test_N = s-test_start
        done_N = cur_test_N*train_N
        t1 = time.time()
        if(done_N!=0):
            print("(%d/%d): est: %.2f h" % (done_N, total_N, (total_N-done_N)/(done_N/(t1-t0)) / 60 / 60))

        dict_tree={}
        for test_id in dict_features_test:  # 1
            test_feature = dict_features_test[test_id]
            locations_1 = test_feature['loc']
            descriptors_1 = test_feature['des']
            d1_tree = cKDTree(descriptors_1, leafsize=50)

            dict_tree[test_id] = (d1_tree, locations_1)

        for t in range(0, len(train_files), train_batch_size):
            t_end = min(t + train_batch_size,len(train_files))
            print("==train processing...(%d-%d)/%d" % (t, t_end, len(train_files)))

            dict_features_train = extract_features(train_dir_name, train_files, t, t_end)

            for test_id in dict_features_test:  # 1
                (d1_tree, locations_1) = dict_tree[test_id]
                num_features_1 = locations_1.shape[0]

                for train_id in dict_features_train:  # 2
                    train_feature = dict_features_train[train_id]
                    locations_2 = train_feature['loc']
                    descriptors_2 = train_feature['des']
                    num_features_2 = locations_2.shape[0]

                    _, indices = d1_tree.query(
                            descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

                    locations_1_to_use = np.array([
                                                          locations_1[indices[i],]
                                                          for i in range(num_features_2)
                                                          if indices[i] != num_features_1
                                                          ])

                    inliers_sum = locations_1_to_use.shape[0]
                    if (inliers_sum > _INLIERS_THRESHOLD):
                        #tf.logging.info("[%s-%s]: shape: %d" % (train_id, test_id, inliers_sum))
                        if test_id not in cur_out:
                            cur_out[test_id]=[]
                        cur_out[test_id].append((train_id, inliers_sum))

        for test_id in cur_out:
            cur_out_data = cur_out[test_id]
            out_file_path = out_dir + "/" + test_id + ".txt"
            with open(out_file_path, 'w') as the_file:
                for (train_id, inliers_sum) in cur_out_data:
                    the_file.write("%s,%s\n"% (train_id, inliers_sum))


if __name__ == '__main__':
    t0 = datetime.datetime.now()
    main()
    print("done.")
    print(datetime.datetime.now()-t0)
