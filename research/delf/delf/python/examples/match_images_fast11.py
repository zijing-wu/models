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

#import annoy
from annoy import AnnoyIndex
#import random

cmd_args = None

_DEBUG = True
_DISTANCE_THRESHOLD = 0.8
_LOAD_FILE_PROCESSOR = 4
_QUERY_PROCESSOR = 4
_TEST_FILE_NUM_START = 0
_TEST_FILE_NUM_END = 100000000
_FEATURE_DS = 1
_PCA_DIM = 40
_TREE_NUM = 8
_KNN_K = 10

_REBUILD_TREE = True
_TREE_SAVE_FILE = 'annoy_tree_ds8_ds1.ann'

_FEATURE_SIZE = int(_PCA_DIM/_FEATURE_DS)
PLOT_FIG = False

g_files=[]
g_dir_name=""
g_t0=None
def f(i, i_start, i_end, descriptors, cur_idx, idx_arr, label_arr, lock):

    if (i % 1000 == 0):
        t1 = time.time()
        cur_n = len(idx_arr)
        total_n = len(g_files)
        dt = t1-g_t0
        if(dt != 0 and cur_n!=0):
            print("loading features...(%d/%d), est: %.2f m" % (cur_n, total_n, (total_n-cur_n)/(cur_n/dt)/60))
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
    ret = [label_arr[index] for index in idxs]
    return ret

annoy_tree, descriptors_query_test = None, None
def f2(i, cur_idx, lock, M):
    cur_idx_value = cur_idx.value
    if (i % 1000 == 0):
        t1 = time.time()
        total_n = M.value
        dt = t1-g_t0
        if(dt != 0 and cur_idx_value!=0):
            print("knn query...(%d/%d), est: %.2f m" % (cur_idx_value, total_n, (total_n-cur_idx_value)/(cur_idx_value/dt)/60))
        sys.stdout.flush()

    v = descriptors_query_test[i]
    idx = annoy_tree.get_nns_by_vector(v, _KNN_K, search_k=-1, include_distances=False)
    with lock:
        cur_idx.value += 1
    return idx

def main():
    global annoy_tree, descriptors_query_test, g_indices

    PKL_FILE_TRAIN = 'save_train'
    PKL_FILE_TEST = 'save_test'

    if(_DEBUG):
        train_dir, test_dir, out_dir, loadtrain, loadtest = (
            'ox_train_features_ds8', 'ox_train_features_ds8', 'lines_out_32_2', 'n', 'n')
        train_dir_name = os.path.abspath(train_dir)
        train_files = [f for f in os.listdir(train_dir_name) if isfile(join(train_dir_name, f))]
    else:
        if(True):
            if len(sys.argv) != 7:
                print('Syntax: {} <train_file_list> <train_dir/> <test_dir/> <out_dir/> <train_save> <test_save>\n\
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
        print("saving...[%s]" % (PKL_FILE_TRAIN))
        np.savez(PKL_FILE_TRAIN, descriptors_list_train=descriptors_list_train,
                 label_arr_train=label_arr_train, idx_arr_train=idx_arr_train)
    else:
        print("loading...[%s]" % (loadtrain))
        npzfile = np.load(loadtrain+'.npz')
        if(_REBUILD_TREE):
            descriptors_list_train = npzfile['descriptors_list_train']
        label_arr_train = npzfile['label_arr_train']
        idx_arr_train = npzfile['idx_arr_train']

    if (loadtest == 'n'):
        descriptors_query_test, label_arr_test, idx_arr_test = \
            extract_features_aggregate_mul(test_dir_name, test_files, _TEST_FILE_NUM_START, min(_TEST_FILE_NUM_END, len(test_files)))
        print("saving...[%s]" % (PKL_FILE_TEST))
        np.savez(PKL_FILE_TEST, descriptors_query_test=descriptors_query_test,
                 label_arr_test=label_arr_test, idx_arr_test=idx_arr_test)
    else:
        print("loading...[%s]" % (loadtest))
        npzfile = np.load(loadtest+'.npz')
        descriptors_query_test = npzfile['descriptors_query_test']
        label_arr_test = npzfile['label_arr_test']
        idx_arr_test = npzfile['idx_arr_test']

    feature_size = _FEATURE_SIZE
    if(_REBUILD_TREE):
        print("building tree...")
        annoy_tree = AnnoyIndex(feature_size, metric='euclidean') #"angular", "euclidean", "manhattan", or "hamming"
        for i in range(descriptors_list_train.shape[0]):
            annoy_tree.add_item(i, descriptors_list_train[i])
        annoy_tree.build(_TREE_NUM)
        print("saving tree...[%s]" % (_TREE_SAVE_FILE))
        annoy_tree.save(_TREE_SAVE_FILE)
    else:
        print("loading tree...[%s]"%(_TREE_SAVE_FILE))
        annoy_tree = AnnoyIndex(feature_size, metric='euclidean')
        annoy_tree.load(_TREE_SAVE_FILE)  # super fast, will just mmap the file

    M = descriptors_query_test.shape[0]
    print("query size:%d,%d" % (descriptors_query_test.shape[0], descriptors_query_test.shape[1]))
    t0 = datetime.datetime.now()
    sys.stdout.flush()

    with Pool(processes=_QUERY_PROCESSOR) as pool:
        with Manager() as manager:
            cur_idx = manager.Value('i', 0)
            lock = manager.Lock()
            indices = pool.starmap(f2, product(range(0, M), [cur_idx], [lock], [manager.Value('i', M)]))

    print("query time:")
    print(datetime.datetime.now() - t0)
    print("indices size:"+str(len(indices)))

    start_j=0
    prev_end_j=0
    data_lines={}
    for i in range(len(idx_arr_test)):
        if (i % 1000 == 0):
            print("processing output...(%d/%d)" % (i, len(idx_arr_test)))
        test_id = label_arr_test[i]
        end_j = idx_arr_test[i]
        cur_lines={}
        for j in range(start_j, end_j): # for each feature j in test i
            train_ids = idx2label(indices[j], label_arr_train, idx_arr_train)
            for train_id in train_ids:
                if train_id not in cur_lines:
                    cur_lines[train_id]=0
                cur_lines[train_id] += 1

        data_lines[test_id] = cur_lines

        start_j += (end_j-prev_end_j)
        prev_end_j=end_j

    out_file_path = out_dir
    print("saving output...[%s.npz]" % (out_file_path))
    np.savez(out_file_path, data_lines=data_lines)


if __name__ == '__main__':
    t0 = datetime.datetime.now()
    main()
    print("done.")
    print(datetime.datetime.now()-t0)
