#!/bin/bash

dir_name=(
    "bcp1_6000-bcp2_6000-exp0013" \
    "bcp2_6000-bcp3_6000-exp0013" \
    "bcp1_6000-pcp1_6000-exp0013" \
    "bcp1_6000-mp1_6000-exp0013" \
    "bcp2_6000-bcp1_6000-exp0013" \
    "bcp3_6000-bcp1_6000-exp0013" \
    "bcp3_6000-bcp2_6000-exp0013" \
    "pcp1_6000-bcp1_6000-exp0013" \
    "mp1_6000-bcp1_6000-exp0013" \
    "bcp1_6000-bcp3_6000-exp0013" \
    "celseq_10xv3" \
    "celseq_dropseq" \
    "celseq_indrop" \
    "celseq_seqwell" \
    "celseq_smartseq" \
    "10xv3_celseq" \
    "10xv3_dropseq" \
    "10xv3_indrop" \
    "10xv3_seqwell" \
    "10xv3_smartseq" \
    "dropseq_celseq" \
    "dropseq_10xv3" \
    "dropseq_indrop" \
    "dropseq_seqwell" \
    "dropseq_smartseq" \
    "indrop_celseq" \
    "indrop_10xv3" \
    "indrop_dropseq" \
    "indrop_seqwell" \
    "indrop_smartseq" \
    "seqwell_celseq" \
    "seqwell_10xv3" \
    "seqwell_dropseq" \
    "seqwell_indrop" \
    "seqwell_smartseq" \
    "smartseq_celseq" \
    "smartseq_10xv3" \
    "smartseq_dropseq" \
    "smartseq_indrop" \
    "smartseq_seqwell"
    
)

projs=(
    "bcp1_6000-bcp2_6000" \
    "bcp2_6000-bcp3_6000" \
    "bcp1_6000-pcp1_6000" \
    "bcp1_6000-mp1_6000" \
    "bcp2_6000-bcp1_6000" \
    "bcp3_6000-bcp1_6000" \
    "bcp3_6000-bcp2_6000" \
    "pcp1_6000-bcp1_6000" \
    "mp1_6000-bcp1_6000" \
    "bcp1_6000-bcp3_6000" \
    "celseq_10xv3" \
    "celseq_dropseq" \
    "celseq_indrop" \
    "celseq_seqwell" \
    "celseq_smartseq" \
    "10xv3_celseq" \
    "10xv3_dropseq" \
    "10xv3_indrop" \
    "10xv3_seqwell" \
    "10xv3_smartseq" \
    "dropseq_celseq" \
    "dropseq_10xv3" \
    "dropseq_indrop" \
    "dropseq_seqwell" \
    "dropseq_smartseq" \
    "indrop_celseq" \
    "indrop_10xv3" \
    "indrop_dropseq" \
    "indrop_seqwell" \
    "indrop_smartseq" \
    "seqwell_celseq" \
    "seqwell_10xv3" \
    "seqwell_dropseq" \
    "seqwell_indrop" \
    "seqwell_smartseq" \
    "smartseq_celseq" \
    "smartseq_10xv3" \
    "smartseq_dropseq" \
    "smartseq_indrop" \
    "smartseq_seqwell"
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