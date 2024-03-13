#!/bin/bash

projects=(
          "bcp1_6000-bcp2_6000-exp0013"
          "bcp1_6000-bcp2_6000-exp0050" \
          "bcp1_6000-bcp2_6000-exp0047" \
          "bcp1_6000-bcp3_6000-exp0013" \
          "bcp1_6000-bcp3_6000-exp0050" \
          "bcp1_6000-bcp3_6000-exp0047" \
          "bcp2_6000-bcp3_6000-exp0013" \
          "bcp2_6000-bcp3_6000-exp0047" \
          "bcp2_6000-bcp3_6000-exp0050" \
          "bcp1_6000-pcp1_6000-exp0013" \
          "bcp1_6000-pcp1_6000-exp0038" \
          "bcp1_6000-pcp1_6000-exp0050" \
          "bcp1_6000-pcp1_6000-exp0047" \
          "bcp1_6000-mp1_6000-exp0013" \
          "bcp1_6000-mp1_6000-exp0040" \
          "bcp1_6000-mp1_6000-exp0050" \
          "bcp1_6000-mp1_6000-exp0047" \
          "bcp2_6000-bcp1_6000-exp0013" \
          "bcp2_6000-bcp1_6000-exp0050" \
          "bcp2_6000-bcp1_6000-exp0047" \
          "bcp3_6000-bcp1_6000-exp0013" \
          "bcp3_6000-bcp1_6000-exp0050" \
          "bcp3_6000-bcp1_6000-exp0047" \
          "bcp3_6000-bcp2_6000-exp0013" \
          "bcp3_6000-bcp2_6000-exp0047" \
          "bcp3_6000-bcp2_6000-exp0050" \
          "pcp1_6000-bcp1_6000-exp0013" \
          "pcp1_6000-bcp1_6000-exp0038" \
          "pcp1_6000-bcp1_6000-exp0050" \
          "pcp1_6000-bcp1_6000-exp0047" \
          "mp1_6000-bcp1_6000-exp0013" \
          "mp1_6000-bcp1_6000-exp0040" \
          "mp1_6000-bcp1_6000-exp0050" \
          "mp1_6000-bcp1_6000-exp0047"                    
          )

projects2=(
          "bcp1_6000-bcp2_6000-exp0013" \
          "bcp1_6000-bcp3_6000-exp0013" \
          "bcp2_6000-bcp3_6000-exp0013" \
          "bcp1_6000-pcp1_6000-exp0013" \
          "bcp1_6000-mp1_6000-exp0013" \
          "bcp2_6000-bcp1_6000-exp0013" \
          "bcp3_6000-bcp1_6000-exp0013" \
          "bcp3_6000-bcp2_6000-exp0013" \
          "pcp1_6000-bcp1_6000-exp0013" \
          "mp1_6000-bcp1_6000-exp0013"
)


for i in ${projects[@]}; do
    # ref_query_auxilary
    CUDA_VISIBLE_DEVICES=0 python gt_main.py --adj_training --active_learning --data_dir=./experiments/$i --use_auxilary
    CUDA_VISIBLE_DEVICES=0 python gt_main.py --adj_training --data_dir=./experiments/$i --use_auxilary
    CUDA_VISIBLE_DEVICES=0 python gt_main.py --active_learning --data_dir=./experiments/$i --use_auxilary
    CUDA_VISIBLE_DEVICES=0 python gt_main.py --data_dir=./experiments/$i --use_auxilary
        
done

for i in ${projects2[@]}; do
    # ref_query
    CUDA_VISIBLE_DEVICES=0 python gt_main.py --adj_training --active_learning --data_dir=./experiments/$i
    CUDA_VISIBLE_DEVICES=0 python gt_main.py --adj_training --data_dir=./experiments/$i
    CUDA_VISIBLE_DEVICES=0 python gt_main.py --active_learning --data_dir=./experiments/$i
    CUDA_VISIBLE_DEVICES=0 python gt_main.py --data_dir=./experiments/$i
done