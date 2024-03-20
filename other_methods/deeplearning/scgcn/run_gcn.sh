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
    "celseq-10xv3" \
    "celseq-dropseq" \
    "celseq-indrop" \
    "celseq-seqwell" \
    "celseq-smartseq" \
    "10xv3-celseq" \
    "10xv3-dropseq" \
    "10xv3-indrop" \
    "10xv3-seqwell" \
    "10xv3-smartseq" \
    "dropseq-celseq" \
    "dropseq-10xv3" \
    "dropseq-indrop" \
    "dropseq-seqwell" \
    "dropseq-smartseq" \
    "indrop-celseq" \
    "indrop-10xv3" \
    "indrop-dropseq" \
    "indrop-seqwell" \
    "indrop-smartseq" \
    "seqwell-celseq" \
    "seqwell-10xv3" \
    "seqwell-dropseq" \
    "seqwell-indrop" \
    "seqwell-smartseq" \
    "smartseq-celseq" \
    "smartseq-10xv3" \
    "smartseq-dropseq" \
    "smartseq-indrop" \
    "smartseq-seqwell" \
    "mouse-human" \
    "human-mouse" \
    "mouse-humancombine" \
    "humancombine-mouse"
    
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
    "celseq-10xv3" \
    "celseq-dropseq" \
    "celseq-indrop" \
    "celseq-seqwell" \
    "celseq-smartseq" \
    "10xv3-celseq" \
    "10xv3-dropseq" \
    "10xv3-indrop" \
    "10xv3-seqwell" \
    "10xv3-smartseq" \
    "dropseq-celseq" \
    "dropseq-10xv3" \
    "dropseq-indrop" \
    "dropseq-seqwell" \
    "dropseq-smartseq" \
    "indrop-celseq" \
    "indrop-10xv3" \
    "indrop-dropseq" \
    "indrop-seqwell" \
    "indrop-smartseq" \
    "seqwell-celseq" \
    "seqwell-10xv3" \
    "seqwell-dropseq" \
    "seqwell-indrop" \
    "seqwell-smartseq" \
    "smartseq-celseq" \
    "smartseq-10xv3" \
    "smartseq-dropseq" \
    "smartseq-indrop" \
    "smartseq-seqwell" \
    "mouse-human" \
    "human-mouse" \
    "mouse-humancombine" \
    "humancombine-mouse"
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