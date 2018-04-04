#!/bin/bash
N=1218618
dt=60

n0=($(ls -l train_features_ds2/ | wc -l))
#n0=ls -l train_features/ | wc -l
echo "cur train features files: $n0"

echo "sleep $dt s"
sleep $dt
n1=($(ls -l train_features_ds2/ | wc -l))
echo "cur train features files: $n1"


dn=$(($n1-$n0))
v=$(($dn/$dt))
t=$(($N/$v/60/60))
echo "est:$t h"


