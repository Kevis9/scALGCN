#!/bin/bash

dir_name=(
    "mp1_6000-pcp1_6000-exp0050" \
    "pcp1_6000-mp1_6000-exp0050"     
)

projs=(
    "mp1_6000-pcp1_6000" \
    "pcp1_6000-mp1_6000"
)

# 获取数组长度
length=${#projs[@]}
export rgraph=FALSE
export basepath="/home/hwl/scALGCN/experiments"

for ((i = 0; i < length; i++)); do
    python h5_adaptor.py --dir_name=/home/hwl/scALGCN/experiments/${dir_name[i]} --proj=${projs[i]}
    Rscript data_preprocess.R $basepath/${dir_name[i]}/raw_data $rgraph
    python train_for_compare.py --proj ${projs[i]}    # 这里是tensorflow的参数传递方式不需要=
done