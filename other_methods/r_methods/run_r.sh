#!/bin/bash

dir_name=(
    # "bcp1_6000-bcp2_6000-exp0013" \
    # "bcp2_6000-bcp3_6000-exp0013" \
    # "bcp1_6000-pcp1_6000-exp0013" \
    # "bcp1_6000-mp1_6000-exp0013" \
    # "bcp2_6000-bcp1_6000-exp0013" \
    # "bcp3_6000-bcp1_6000-exp0013" \
    # "bcp3_6000-bcp2_6000-exp0013" \
    # "pcp1_6000-bcp1_6000-exp0013" \
    # "mp1_6000-bcp1_6000-exp0013" \
    "mp1_6000-pcp1_6000-exp0050" \
    "pcp1_6000-mp1_6000-exp0050"    
)

projs=(
    # "bcp1_6000-bcp2_6000" \
    # "bcp2_6000-bcp3_6000" \
    # "bcp1_6000-pcp1_6000" \
    # "bcp1_6000-mp1_6000" \
    # "bcp2_6000-bcp1_6000" \
    # "bcp3_6000-bcp1_6000" \
    # "bcp3_6000-bcp2_6000" \
    # "pcp1_6000-bcp1_6000" \
    # "mp1_6000-bcp1_6000" \
    "mp1_6000-pcp1_6000" \
    "pcp1_6000-mp1_6000"    
)

# 获取数组长度
length=${#projs[@]}

for ((i = 0; i < length; i++)); do
   python h5_adaptor.py --dir_name=/home/hwl/scALGCN/experiments/${dir_name[i]} --proj=${projs[i]}
   Rscript r_methods.R /home/hwl/scALGCN/experiments/${dir_name[i]} ${projs[i]}
   python decoder.py --proj=${projs[i]}
done