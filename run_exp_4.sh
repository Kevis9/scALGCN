#!/bin/bash

projects=(        
        #   "bcp1_6000-bcp3_6000-exp0013" \
          "bcp2_6000-bcp3_6000-exp0013" \
          "bcp2_6000-bcp3_6000-exp0050" \
          )




for i in ${projects[@]}; do
    # ref_query_auxilary
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --data_dir=./experiments/$i  --use_auxilary
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --add_pos_enc --data_dir=./experiments/$i --use_auxilary
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --active_learning --data_dir=./experiments/$i --use_auxilary
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --adj_training --data_dir=./experiments/$i --use_auxilary
    CUDA_VISIBLE_DEVICES=1 python gt_main.py --adj_training --add_pos_enc --active_learning --data_dir=./experiments/$i --use_auxilary
    
    # ref_query
    # CUDA_VISIBLE_DEVICES=1 python gt_main.py --data_dir=./experiments/$i
    # CUDA_VISIBLE_DEVICES=1 python gt_main.py --add_pos_enc --data_dir=./experiments/$i
    # CUDA_VISIBLE_DEVICES=1 python gt_main.py --active_learning --data_dir=./experiments/$i
    # CUDA_VISIBLE_DEVICES=1 python gt_main.py --adj_training --data_dir=./experiments/$i
    # CUDA_VISIBLE_DEVICES=1 python gt_main.py --adj_training --add_pos_enc --active_learning --data_dir=./experiments/$i       
done