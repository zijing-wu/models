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
import csv

import numpy as np
from scipy.spatial import cKDTree
import tensorflow as tf

from delf import feature_io
from os.path import isfile, join
import datetime
import time
from multiprocessing import Pool, TimeoutError

import os

from multiprocessing import Process, Manager, Lock
from itertools import product
import threading
import pickle

cmd_args = None

_DEBUG = False
_DISTANCE_THRESHOLD = 0.8
_LOAD_FILE_PROCESSOR = 180
_QUERY_PROCESSOR = 188
_TEST_FILE_NUM_START = 0
_TEST_FILE_NUM_END = 1000
_FEATURE_DS = 4

PLOT_FIG = False

g_files=[]
g_dir_name=""
g_t0=None
def f(i, i_start, i_end, descriptors, cur_idx, idx_arr, label_arr, lock):

    if (i % 1000 == 0):
        t1 = time.time()
        cur_n = len(idx_arr)
        total_n = len(g_files)
        print("loading features...(%d/%d), est: %.2f m" % (cur_n, total_n, (total_n-cur_n)/(cur_n/(t1-g_t0))/60))
        sys.stdout.flush()
    #print("files size:%d"%(len(g_files)))
    cur_features_file = g_dir_name + '/' + g_files[i]
    basename = os.path.basename(cur_features_file)
    basename = os.path.splitext(basename)[0]
    try:
        _, _, descriptors_1, _, _ = feature_io.ReadFromFile(cur_features_file)
        #print("descriptors_1 size:%d,%d" % (descriptors_1.shape[0], descriptors_1.shape[1]))
        descriptors_1_ = descriptors_1[:,0:int(descriptors_1.shape[1]/_FEATURE_DS)]
        #print("cut descriptors_1 size:%d,%d" % (descriptors_1_.shape[0], descriptors_1_.shape[1]))
    except:
        print("load feature get error, skip...[%s]" % (cur_features_file))
        return

    with lock:
        #print("descriptors size:%d"%(len(descriptors)))
        descriptors.append(descriptors_1_)
        cur_idx.value += descriptors_1_.shape[0]
        idx_arr.append(cur_idx.value)
        label_arr.append(basename)


def extract_features_aggregate_mul(dir_name, files, i_start, i_end):
    global g_files, g_dir_name, g_cur_idx, _LOAD_FILE_PROCESSOR, g_t0
    g_files = files
    g_dir_name = dir_name
    g_cur_idx = 0
    descriptors_ = []
    idx_arr_=[]
    label_arr_=[]
    g_t0 = time.time()

    with Pool(processes=_LOAD_FILE_PROCESSOR) as pool:
        with Manager() as manager:
            idx_arr = manager.list()
            label_arr = manager.list()
            cur_idx = manager.Value('i', 0)
            descriptors = manager.list()
            lock = manager.Lock()

            pool.starmap(f, product(range(i_start, i_end),[i_start],[i_end], [descriptors], [cur_idx], [idx_arr], [label_arr], [lock]))

            for i in range(len(descriptors)):
                descriptors_.append(descriptors[i])
                idx_arr_.append(idx_arr[i])
                label_arr_.append(label_arr[i])

    if(len(descriptors_)==0 or descriptors_ is None): return None,[],[]
    descriptors_ = np.vstack(descriptors_)


    print("loading features done. size:%d,%d" % (descriptors_.shape[0], descriptors_.shape[1]))

    return descriptors_,label_arr_,idx_arr_

def idx2label(idx, label_arr, idx_arr):
    idxs = np.searchsorted(idx_arr, idx)
    return label_arr[idxs]

def main():
    PKL_FILE_TRAIN = 'save_train.pkl'
    PKL_FILE_TEST = 'save_test.pkl'

    if(_DEBUG):
        train_dir, test_dir, out_dir, loadtrain, loadtest = (
        'ox_train_features/train', 'ox_train_features/test', 'lines_out_3', 'n', 'n')
        train_dir_name = os.path.abspath(train_dir)
        train_files = [f for f in os.listdir(train_dir_name) if isfile(join(train_dir_name, f))]
    else:
        if(True):
            if len(sys.argv) != 7:
                print('Syntax: {} <train_file_list> <train_dir/> <test_dir/> <out_dir/> <load_pkl> <save_pkl>\n\
                      set pkl file to n if load from delf'.format(sys.argv[0]))
                sys.exit(0)
            (train_file_list, train_dir, test_dir, out_dir, loadtrain, loadtest) = sys.argv[1:]
            reader = csv.reader(open(train_file_list, "r"), delimiter=",")

            train_files=[]
            for row in reader:
                train_id = row[0]
                #class_label = row[1]
                cur_path = "%s.delf"%(train_id)
                train_files.append(cur_path)
        else:
            if len(sys.argv) != 6:
                print('Syntax: {} <train_dir/> <test_dir/> <out_dir/> <load_pkl> <save_pkl>\n\
                              set pkl file to n if load from delf'.format(sys.argv[0]))
                sys.exit(0)
            (train_dir, test_dir, out_dir, loadtrain, loadtest) = sys.argv[1:]
            train_dir_name = os.path.abspath(train_dir)
            train_files = [f for f in os.listdir(train_dir_name) if isfile(join(train_dir_name, f))]


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_dir_name = os.path.abspath(train_dir)

    test_dir_name = os.path.abspath(test_dir)
    test_files = [f for f in os.listdir(test_dir_name) if isfile(join(test_dir_name, f))]

    if(loadtrain=='n'):
        descriptors_list_train, label_arr_train, idx_arr_train = \
            extract_features_aggregate_mul(train_dir_name, train_files, 0, len(train_files))
        #print("saving...[%s]" % (PKL_FILE_TRAIN))
        #with open(PKL_FILE_TRAIN, 'wb') as f:
        #    pickle.dump([descriptors_list_train, label_arr_train, idx_arr_train], f)
    else:
        print("loading...[%s]" % (loadtrain))
        with open(loadtrain, 'rb') as f:
            descriptors_list_train, label_arr_train, idx_arr_train = pickle.load(f)

    if (loadtest == 'n'):
        descriptors_query_test, label_arr_test, idx_arr_test = \
            extract_features_aggregate_mul(test_dir_name, test_files, _TEST_FILE_NUM_START, min(_TEST_FILE_NUM_END, len(test_files)))
        #print("saving...[%s]" % (PKL_FILE_TEST))
        #with open(PKL_FILE_TEST, 'wb') as f:
        #    pickle.dump([descriptors_query_test, label_arr_test, idx_arr_test], f)
    else:
        print("loading...[%s]" % (loadtest))
        with open(loadtest, 'rb') as f:
            descriptors_query_test, label_arr_test, idx_arr_test = pickle.load(f)

    dk_tree_train = cKDTree(descriptors_list_train, leafsize=1000000000)

    print("query size:%d,%d" % (descriptors_query_test.shape[0], descriptors_query_test.shape[1]))
    t0 = datetime.datetime.now()
    sys.stdout.flush()
    _, indices = dk_tree_train.query(
        descriptors_query_test, p=2, distance_upper_bound=_DISTANCE_THRESHOLD, n_jobs=_QUERY_PROCESSOR)
    print("tree leaf size:%d"%(dk_tree_train.n))
    print("query time:")
    print(datetime.datetime.now() - t0)
    sys.stdout.flush()

    start_j=0
    prev_end_j=0
    for i in range(len(idx_arr_test)):
        test_id = label_arr_test[i]
        end_j = idx_arr_test[i]
        cur_lines={}
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
