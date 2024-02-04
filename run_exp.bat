@echo off
setlocal enabledelayedexpansion

python gt_main.py
python gt_main.py --add_pos_enc
python gt_main.py --active_learning
python gt_main.py --graph_learning
python gt_main.py --adj_training --add_pos_enc --active_learning