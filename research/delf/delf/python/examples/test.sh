#!/bin/bash


gdb -ex r --args python3 match_images_fast10.py image_list_ds8_2.txt train_features_ds8 test_features_ds8 line_ds8_6 save_train save_test

nohup python3.6 -u match_images_fast9.py image_list_ds8_2.txt train_features_ds8 test_features_ds8 out1 save_train_ds8_ds1_bad save_test_ds8_ds1_bad > nohup.log &

kaggle competitions submit -c landmark-recognition-challenge -f submission.csv -m "Message"


cat image_list_test_ds8.txt | xargs cp -t my_test_features/
zip -r my_test_features_ds8.zip my_test_features/

