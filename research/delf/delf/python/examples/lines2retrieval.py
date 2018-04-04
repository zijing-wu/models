import pandas as pd
import glob
import time
import csv
import sys
import os
import numpy as np
from os.path import basename


def main():
   if len(sys.argv) != 4:
      print('Syntax: {} <INPUT_DIR/> <LINES_THRESHOLD> <OUT_FILENAME>'.format(sys.argv[0]))
      sys.exit(0)
   (INPUT_DIR, LINES_THRESHOLD, OUT_NAME) = sys.argv[1:]
   LINES_THRESHOLD=int(LINES_THRESHOLD)

   INPUT_DIR += "/*.txt"

   #INPUT_DIR, LINES_THRESHOLD, OUT_NAME = 'lines_test/*.txt', 20, 'out'

   dict_lines={}
   out_retr={} #test_id->[train_id1, train_id2, ...]
   out_clas={} #test_id->(train_id1, prob)
   files = glob.glob(INPUT_DIR)
   N_files = len(files)
   i=0
   for fle in files:
      if(i%10==0):
         print("processing...[%d/%d]"%(i,N_files))
      i += 1

      test_id = os.path.splitext(basename(fle))[0]

      reader = csv.reader(open(fle, 'r'), delimiter=',')
      cur_id_list=[]
      cur_line_list=[]
      for row in reader:
         train_id = row[0]
         lines_num = int(row[1])
         if(lines_num < LINES_THRESHOLD): continue

         if test_id not in dict_lines:
            dict_lines[test_id] = []
         dict_lines[test_id].append((train_id, lines_num))
         cur_id_list.append(train_id)
         cur_line_list.append(lines_num)

      cur_line_list, cur_id_list = zip(*sorted(zip(cur_line_list, cur_id_list), reverse=True))
      out_retr[test_id] = cur_id_list
      out_clas[test_id] = (cur_id_list[0], cur_line_list[0], cur_line_list[0]/np.sum(cur_line_list))

   OUT_RETR_FILE = OUT_NAME + '_retr.csv'
   OUT_CLAS_FILE = OUT_NAME + '_clas.csv'

   with open(OUT_RETR_FILE, 'w') as the_file:
      the_file.write("id,images\n")
      for test_id in out_retr:
         train_ids = out_retr[test_id]
         row = ' '.join(train_ids)
         the_file.write("%s,%s\n" % (test_id, row))

   with open(OUT_CLAS_FILE, 'w') as the_file:
      the_file.write("id,landmarks\n")
      for test_id in out_clas:
         (train_id, lines, prob) = out_clas[test_id]
         prob=1
         the_file.write("%s,%s %d %.2f\n" % (test_id, train_id, lines, prob))
   print("done.")

if __name__ == '__main__':
    main()