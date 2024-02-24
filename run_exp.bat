@echo off
setlocal enabledelayedexpansion

@REM python gt_main.py
@REM python gt_main.py --add_pos_enc
@REM python gt_main.py --active_learning
@REM python gt_main.py --graph_learning
@REM python gt_main.py --adj_training --add_pos_enc --active_learning

python gt_main.py
python gt_main.py --use_auxilary=False
python gt_main.py --add_pos_enc --use_auxilary=False
python gt_main.py --active_learning --use_auxilary=False
python gt_main.py --graph_learning --use_auxilary=False
python gt_main.py --adj_training --add_pos_enc --active_learning --use_auxilary=False