@echo off
set ab_path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\platform_v2
set des_ab_path=E:\YuAnHuang\kevislin\second_proj\scALGCN\experiment
set project[0]=cel_seq_10x_v3

for /l %%n in (0,1,0) do (
    set ref_path=%ab_path%\!project[%%n]!\raw_data\ref
    set query_path=%ab_path%\!project[%%n]!\raw_data\query
    set des_path=%des_ab_path%\!project[%%n]!
    set des_data_path=%des_path%\data
    set des_raw_data_path=%des_path%\raw_data
    if not exist %des_path% mkdir %des_path%
    if not exist %des_data_path% mkdir %des_data_path%
    if not exist %des_raw_data_path% mkdir %des_raw_data_path%

    copy %ref_path%\data_1.csv %des_raw_data_path%\ref_data.csv
    copy %ref_path%\label_1.csv %des_raw_data_path%\ref_label.csv
    copy %query_path%\data_1.csv %des_raw_data_path%\query_data.csv
    copy %query_path%\label_1.csv %des_raw_data_path%\query_label.csv

)

echo File copy successfully
