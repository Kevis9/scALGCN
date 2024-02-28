#!/bin/bash

projects=(
        #   "bcp1_6000-bcp2_6000-exp0013"
        #   "bcp1_6000-bcp2_6000-exp0050" \
        #   "bcp1_6000-bcp2_6000-exp0047" \
        #   "bcp1_6000-bcp3_6000-exp0013" \
        #   "bcp1_6000-bcp3_6000-exp0050" \
        #   "bcp1_6000-bcp3_6000-exp0047" \
        #   "bcp2_6000-bcp3_6000-exp0013" \
        #   "bcp2_6000-bcp3_6000-exp0047" \
        #   "bcp2_6000-bcp3_6000-exp0050" \
        #   "bcp1_6000-pcp1_6000-exp0013" \
        #   "bcp1_6000-pcp1_6000-exp0038" \
        #   "bcp1_6000-pcp1_6000-exp0050" \
        #   "bcp1_6000-pcp1_6000-exp0047" \
        #   "bcp1_6000-mp1_6000-exp0013" \
        #   "bcp1_6000-mp1_6000-exp0040" \
        #   "bcp1_6000-mp1_6000-exp0050" \
        #   "bcp1_6000-mp1_6000-exp0047" \
          "bcp2_6000-bcp1_6000-exp0013" \
          "bcp2_6000-bcp1_6000-exp0050" \
          "bcp2_6000-bcp1_6000-exp0047" \
          "bcp3_6000-bcp1_6000-exp0013" \
          "bcp3_6000-bcp1_6000-exp0050" \
          "bcp3_6000-bcp1_6000-exp0047" \
          "bcp3_6000-bcp2_6000-exp0013" \
          "bcp3_6000-bcp2_6000-exp0047" \
        #   "bcp3_6000-bcp2_6000-exp0050" \
        #   "pcp1_6000-bcp1_6000-exp0013" \
        #   "pcp1_6000-bcp1_6000-exp0038" \
        #   "pcp1_6000-bcp1_6000-exp0050" \
        #   "pcp1_6000-bcp1_6000-exp0047" \
        #   "mp1_6000-bcp1_6000-exp0013" \
        #   "mp1_6000-bcp1_6000-exp0040" \
        #   "mp1_6000-bcp1_6000-exp0050" \
        #   "mp1_6000-bcp1_6000-exp0047"                    
          )




for i in ${projects[@]}; do
    # ref_query_auxilary
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --data_dir=./experiments/$i  --use_auxilary
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --add_pos_enc --data_dir=./experiments/$i --use_auxilary
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --active_learning --data_dir=./experiments/$i --use_auxilary
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --adj_training --data_dir=./experiments/$i --use_auxilary
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --adj_training --add_pos_enc --active_learning --data_dir=./experiments/$i --use_auxilary
    
    # ref_query
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --data_dir=./experiments/$i
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --add_pos_enc --data_dir=./experiments/$i
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --active_learning --data_dir=./experiments/$i
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --adj_training --data_dir=./experiments/$i
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --adj_training --add_pos_enc --active_learning --data_dir=./experiments/$i       
done