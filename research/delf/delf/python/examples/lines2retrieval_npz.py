import pandas as pd
import glob
import time
import csv
import sys
import os
import numpy as np
from os.path import basename

_DEBUG=True
def main():
   if(_DEBUG):
      (INPUT_LINES_NPZ, LINES_THRESHOLD, TRAIN_CSV, TEST_CSV, OUT_NAME)=("lines_out_32_2.npz", 0, "data_retrieval/train.csv", "data_retrieval/test.csv", "ds2_1wx1w")
   else:
      if len(sys.argv) != 6:
         print('Syntax: {} <INPUT_LINES_NPZ> <LINES_THRESHOLD> <TRAIN_CSV> <TEST_CSV> <OUT_FILENAME>'.format(sys.argv[0]))
         sys.exit(0)
      (INPUT_LINES_NPZ, LINES_THRESHOLD, TRAIN_CSV, TEST_CSV, OUT_NAME) = sys.argv[1:]
      LINES_THRESHOLD=int(LINES_THRESHOLD)

   print("loading...[%s]" % (INPUT_LINES_NPZ))
   npzfile = np.load(INPUT_LINES_NPZ)
   data_lines = npzfile['data_lines'][()]
   label_arr_test = npzfile['label_arr_test'][()]

   reader = csv.reader(open(TEST_CSV, "r"), delimiter=",")
   csv_test_id = []
   n=0
   for row in reader:
      if(n==0):
         n += 1
         continue
      test_id = row[0]
      csv_test_id.append(test_id)
      n += 1

   reader = csv.reader(open(TRAIN_CSV, "r"), delimiter=",")
   csv_train_id2label = {}
   n = 0
   for row in reader:
      if (n == 0):
         n += 1
         continue
      train_id = row[0]
      train_label = row[2]
      csv_train_id2label[train_id] = train_label
      n += 1

   dict_lines={}
   out_retr={} #test_id->[train_id1, train_id2, ...]
   out_clas={} #test_id->(train_id1, prob)

   i=0
   for i in range(len(label_arr_test)):
      if(i%1000==0):
         print("processing...[%d/%d]"%(i,len(data_lines)))

      test_id = label_arr_test[i]
      cur_lines = data_lines[i]

      cur_id_list=[]
      cur_line_list=[]
      for train_id in cur_lines:
         lines_num = cur_lines[train_id]
         if(lines_num < LINES_THRESHOLD): continue


         if test_id not in dict_lines:
            dict_lines[test_id] = []
         dict_lines[test_id].append((train_id, lines_num))
         cur_id_list.append(train_id)
         cur_line_list.append(lines_num)

      if(len(cur_id_list)==0):
          out_retr[test_id] = []
          out_clas[test_id] = None
          continue

      cur_line_list, cur_id_list = zip(*sorted(zip(cur_line_list, cur_id_list), reverse=True))
      out_retr[test_id] = cur_id_list

      prob_list=[]
      total_lines = np.sum(cur_line_list)
      for j in range(len(cur_id_list)):
         prob_list.append(cur_line_list[j]/total_lines)
      out_clas[test_id] = (cur_id_list, cur_line_list, prob_list)

      if(_DEBUG):
         print(test_id)
         print((cur_id_list, cur_line_list, prob_list))

   OUT_RETR_FILE = OUT_NAME + '_retr.csv'
   OUT_CLAS_FILE = OUT_NAME + '_clas.csv'

   with open(OUT_RETR_FILE, 'w') as the_file:
      the_file.write("id,images\n")
      for test_id in csv_test_id:
         if test_id in out_retr:
            train_ids = out_retr[test_id]
            row = ' '.join(train_ids)
         else:
            row = ' '
         the_file.write("%s,%s\n" % (test_id, row))

   with open(OUT_CLAS_FILE, 'w') as the_file:
      the_file.write("id,landmarks\n")
      for test_id in csv_test_id:
         if((test_id in out_clas) and (out_clas[test_id] is not None)):
            (cur_id_list, cur_line_list, prob_list) = out_clas[test_id]
            #the_file.write("%s,%s %.2f\n" % (test_id, csv_train_id2label[cur_id_list[0]], prob_list[0]))
            the_file.write("%s,%s %.2f\n" % (test_id, csv_train_id2label[cur_id_list[0]], 1))
         else:
            the_file.write("%s, \n" % (test_id))

   print("done.")

if __name__ == '__main__':
    main()
