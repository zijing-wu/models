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

_DISTANCE_THRESHOLD = 0.8

PLOT_FIG = False
#INDEX_FEATURE_FOLDER = 'data/oxford5k_features'
#QUERY_FEATURE_FOLDER = 'data/oxford5k_features'
#OUT_PUT_FILE = 'out.csv'




def extract_features(dir_name, files, i_start, i_end):
    dict_features_index = {}
    for i in range(i_start, i_end):
        if(i>2000): break
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

g_files=[]
g_dir_name=""
def f(i, i_start, i_end, descriptors, cur_idx, idx_arr, label_arr, lock):

    if (i % 100 == 0):
        print("loading features...(%d/%d)" % (i, i_end - i_start))
    #print("files size:%d"%(len(g_files)))
    cur_features_file = g_dir_name + '/' + g_files[i]
    basename = os.path.basename(cur_features_file)
    basename = os.path.splitext(basename)[0]
    try:
        _, _, descriptors_1, _, _ = feature_io.ReadFromFile(cur_features_file)
    except:
        print("load feature get error, skip...[%s]" % (cur_features_file))
        return

    with lock:
        #print("descriptors size:%d"%(len(descriptors)))
        descriptors.append(descriptors_1)
        cur_idx.value += descriptors_1.shape[0]
        idx_arr.append(cur_idx.value)
        label_arr.append(basename)


def extract_features_aggregate_mul(dir_name, files, i_start, i_end):
    global g_files, g_dir_name, g_cur_idx
    g_files = files
    g_dir_name = dir_name
    g_cur_idx = 0
    descriptors_ = []
    idx_arr_=[]
    label_arr_=[]


    with Pool(processes=176) as pool:
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

def extract_features_aggregate(dir_name, files, i_start, i_end):
    descriptors = None
    label_arr=[]
    idx_arr=[]
    cur_idx=0
    n=0
    for i in range(i_start, i_end):
        #if(i>100): break
        if(i % 100 == 0):
            print("loading features...(%d/%d)"%(i,i_end-i_start))
        cur_features_file = dir_name + '/' + files[i]
        basename = os.path.basename(cur_features_file)
        basename = os.path.splitext(basename)[0]
        try:
            _, _, descriptors_1, _, _ = feature_io.ReadFromFile(cur_features_file)
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
    if(descriptors is None): return None,[],[]

    descriptors=descriptors.value
    print("loading features done. size:%d,%d" % (descriptors.shape[0], descriptors.shape[1]))
    #list1, list2 = zip(*sorted(zip(idx_arr, label_arr)))

    return descriptors,label_arr,idx_arr

def idx2label(idx, label_arr, idx_arr):
    idxs = np.searchsorted(idx_arr, idx)
    return label_arr[idxs]

_INLIERS_THRESHOLD = 150
_DEBUG=False
def main():
    PKL_FILE_TRAIN = 'save_train.pkl'
    PKL_FILE_TEST = 'save_test.pkl'

    if(_DEBUG):
        train_dir, test_dir, out_dir, loadtrain, loadtest = (
        'ox_train_features/train', 'ox_train_features/test', 'lines_out_2', 'n', 'n')
        train_dir_name = os.path.abspath(train_dir)
        train_files = [f for f in os.listdir(train_dir_name) if isfile(join(train_dir_name, f))]
    else:
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

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

#   tf.logging.set_verbosity(tf.logging.INFO)

    train_dir_name = os.path.abspath(train_dir)

    test_dir_name = os.path.abspath(test_dir)
    test_files = [f for f in os.listdir(test_dir_name) if isfile(join(test_dir_name, f))]

    if(loadtrain=='n'):
        descriptors_list_train, label_arr_train, idx_arr_train = \
            extract_features_aggregate_mul(train_dir_name, train_files, 0, len(train_files))
        print("saving...[%s]" % (PKL_FILE_TRAIN))
        with open(PKL_FILE_TRAIN, 'wb') as f:
            pickle.dump([descriptors_list_train, label_arr_train, idx_arr_train], f)
    else:
        print("loading...[%s]" % (loadtrain))
        with open(loadtrain, 'rb') as f:
            descriptors_list_train, label_arr_train, idx_arr_train = pickle.load(f)

    if (loadtest == 'n'):
        descriptors_query_test, label_arr_test, idx_arr_test = \
            extract_features_aggregate_mul(test_dir_name, test_files, 0, 10000)
        print("saving...[%s]" % (PKL_FILE_TEST))
        with open(PKL_FILE_TEST, 'wb') as f:
            pickle.dump([descriptors_query_test, label_arr_test, idx_arr_test], f)
    else:
        print("loading...[%s]" % (loadtest))
        with open(loadtest, 'rb') as f:
            descriptors_query_test, label_arr_test, idx_arr_test = pickle.load(f)

    dk_tree_train = cKDTree(descriptors_list_train, leafsize=1000000000)

    print("query size:%d,%d" % (descriptors_query_test.shape[0], descriptors_query_test.shape[1]))
    t0 = datetime.datetime.now()
    _, indices = dk_tree_train.query(
        descriptors_query_test, p=2, distance_upper_bound=_DISTANCE_THRESHOLD, n_jobs=176)
    print("tree leaf size:%d"%(dk_tree_train.n))
    print("query time:")
    print(datetime.datetime.now() - t0)

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
