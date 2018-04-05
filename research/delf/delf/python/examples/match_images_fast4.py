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

import numpy as np
from scipy.spatial import cKDTree
import tensorflow as tf

from delf import feature_io
from os.path import isfile, join
import datetime
import time

import os

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

def extract_features_aggregate(dir_name, files, i_start, i_end):
    descriptors = None
    label_arr=[]
    idx_arr=[]
    cur_idx=0
    n=0
    for i in range(i_start, i_end):
        if(i % 100 == 0):
            print("loading features...(%d/%d)"%(i,i_end-i_start))
        cur_features_file = dir_name + '/' + files[i]
        basename = os.path.basename(cur_features_file)
        basename = os.path.splitext(basename)[0]
        try:
            _, _, descriptors_1, _, _ = feature_io.ReadFromFile(
                cur_features_file)
        except:
            print("load feature get error, skip...[%s]"%(cur_features_file))
            continue
        if(descriptors is None):
            descriptors = descriptors_1
        else:
            #print("descriptors size:%d,%d" % (descriptors.shape[0], descriptors.shape[1]))
            #print("descriptors_1 size:%d,%d" % (descriptors_1.shape[0], descriptors_1.shape[1]))
            descriptors = np.concatenate((descriptors, descriptors_1), axis=0)
        #descriptors.append(descriptors_1)

        cur_idx+=descriptors_1.shape[0]
        idx_arr.append(cur_idx)
        label_arr.append(basename)
        n+=1

    print("loading features done. size:%d,%d" % (descriptors.shape[0], descriptors.shape[1]))
    return descriptors,label_arr,idx_arr

def build_kdtree(descriptors_list):
    tree=None
    d1_tree = cKDTree(descriptors_list, leafsize=50)
    return tree

def idx2label(idx, label_arr, idx_arr):
    idxs = np.searchsorted(idx_arr, idx)
    return label_arr[idxs]

_INLIERS_THRESHOLD = 150
def main():
    if len(sys.argv) != 5:
        print('Syntax: {} <train_dir/> <test_dir/> <batch_size> <out_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (train_dir, test_dir, train_batch_size, out_dir) = sys.argv[1:]
    #(train_dir, test_dir, train_batch_size, out_dir) = ('ox_train_features/','ox_train_features/',1000,'lines_out')
    #test_start=int(test_start)
    #test_end=int(test_end)
    #test_batch_size=int(test_batch_size)
    train_batch_size = int(train_batch_size)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tf.logging.set_verbosity(tf.logging.INFO)

    train_dir_name = os.path.abspath(train_dir)
    train_files = [f for f in os.listdir(train_dir_name) if isfile(join(train_dir_name, f))]
    test_dir_name = os.path.abspath(test_dir)
    test_files = [f for f in os.listdir(test_dir_name) if isfile(join(test_dir_name, f))]

    #test_N = test_end-test_start
    #train_N = len(train_files)
    #total_N = test_N*train_N
    t0 = time.time()

    descriptors_query_test, label_arr_test, idx_arr_test = extract_features_aggregate(test_dir_name, test_files, 0, len(test_files))

    for t in range(0, len(train_files), train_batch_size):
        t_end = min(t + train_batch_size, len(train_files))
        descriptors_list_train, label_arr_train, idx_arr_train = extract_features_aggregate(train_dir_name, train_files, t, t_end)
        dk_tree_train = cKDTree(descriptors_list_train, leafsize=50)

        print("query size:%d,%d" % (descriptors_query_test.shape[0], descriptors_query_test.shape[1]))
        _, indices = dk_tree_train.query(
            descriptors_query_test, distance_upper_bound=_DISTANCE_THRESHOLD)

        start_j=0
        prev_end_j=0
        for i in range(len(idx_arr_test)):
            test_id = label_arr_train[i]
            end_j = idx_arr_test[i]
            cur_lines={}
            #print(indices[start_j:end_j])
            skip_num=0
            for j in range(start_j, end_j): # for each feature j in test i
                if indices[j] != dk_tree_train.n:
                    train_id = idx2label(indices[j], label_arr_train, idx_arr_train)
                    if train_id not in cur_lines:
                        cur_lines[train_id]=0
                    cur_lines[train_id] += 1
                else:
                    skip_num+=1
            #print("size:%d; skip:%d\n"%(len(cur_lines),skip_num))

            start_j += (end_j-prev_end_j)
            prev_end_j=end_j

            out_file_path = out_dir + "/" + test_id + ".txt"
            with open(out_file_path, 'w') as the_file:
                for train_id in cur_lines:
                    inliers_sum = cur_lines[train_id]
                    the_file.write("%s,%s\n" % (train_id, inliers_sum))


if __name__ == '__main__':
    t0 = datetime.datetime.now()
    main()
    print("done.")
    print(datetime.datetime.now()-t0)
