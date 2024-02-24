#!/bin/bash

# projects=("bcp1_6000-bcp2_6000-exp0013" \
#           "bcp1_6000-bcp2_6000-exp0050" \
#           "bcp1_6000-bcp2_6000-exp0047" \
#           "bcp1_6000-bcp3_6000-exp0013" \
#           "bcp1_6000-bcp3_6000-exp0050" \
#           "bcp1_6000-bcp3_6000-exp0047" \
#           "bcp2_6000-bcp3_6000-exp0013" \
#           "bcp2_6000-bcp3_6000-exp0047" \
#           "bcp2_6000-bcp3_6000-exp0050" \
#           "bcp1_6000-pcp1_6000-exp0013" \
#           "bcp1_6000-pcp1_6000-exp0038" \
#           "bcp1_6000-pcp1_6000-exp0050" \
#           "bcp1_6000-pcp1_6000-exp0047" \
#           "bcp1_6000-mp1_6000-exp0013" \
#           "bcp1_6000-mp1_6000-exp0040" \
#           "bcp1_6000-mp1_6000-exp0050" \
#           "bcp1_6000-mp1_6000-exp0047")

projects=("bcp1_6000-bcp2_6000-exp0013")

for i in $projects; do
    # ref_query_auxilary
    # python gt_main.py --data_dir=./experiments/$i  --use_auxilary
    # python gt_main.py --add_pos_enc --data_dir=./experiments/$i --use_auxilary
    # python gt_main.py --active_learning --data_dir=./experiments/$i --use_auxilary
    python gt_main.py --adj_training --data_dir=./experiments/$i --use_auxilary
    # python gt_main.py --adj_training --add_pos_enc --active_learning --data_dir=./experiments/$i --use_auxilary
    
    # ref_query
    # python gt_main.py --data_dir=./experiments/$i
    # python gt_main.py --add_pos_enc --data_dir=./experiments/$i
    # python gt_main.py --active_learning --data_dir=./experiments/$i
    # python gt_main.py --adj_training --data_dir=./experiments/$i
    # python gt_main.py --adj_training --add_pos_enc --active_learning --data_dir=./experiments/$i
    
    # # query_ref_auxilary
    # python gt_main.py --data_dir=./experiments/$i --exp_reverse --use_auxilary
    # python gt_main.py --add_pos_enc --data_dir=./experiments/$i --exp_reverse --use_auxilary
    # python gt_main.py --active_learning --data_dir=./experiments/$i --exp_reverse --use_auxilary
    # python gt_main.py --adj_training --data_dir=./experiments/$i --exp_reverse --use_auxilary
    # python gt_main.py --adj_training --add_pos_enc --active_learning --data_dir=./experiments/$i --exp_reverse --use_auxilary

    # # query_ref
    # python gt_main.py --data_dir=./experiments/$i --exp_reverse
    # python gt_main.py --add_pos_enc --data_dir=./experiments/$i --exp_reverse
    # python gt_main.py --active_learning --data_dir=./experiments/$i --exp_reverse
    # python gt_main.py --adj_training --data_dir=./experiments/$i --exp_reverse
    # python gt_main.py --adj_training --add_pos_enc --active_learning --data_dir=./experiments/$i --exp_reverse
    
done

