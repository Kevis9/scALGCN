@echo off
setlocal enabledelayedexpansion
set ab_path=E:\YuAnHuang\kevislin\Cell_Classification\experiment\omic
set des_ab_path=E:\YuAnHuang\kevislin\second_proj\scALGCN\experiment
set projects=A549,^
kidney,^
lung
@REM 10x_v3_drop_seq,^
@REM 5061_84133,^
@REM 84133_5061,^
@REM 84133_81608,^
@REM 84133_85241,^
@REM 84133_combine,^
@REM drop_seq_10x_v3,^
@REM drop_seq_seq_well,^
@REM indrop_10x_v3,^
@REM indrop_drop_seq,^
@REM seq_well_10x_v3,^
@REM seq_well_drop_seq

for %%i in (%projects%) do (
    set ref_path=!ab_path!\%%i\raw_data\ref
    set query_path=!ab_path!\%%i\raw_data\query
    set des_path=!des_ab_path!\%%i
    set des_data_path=!des_path!\data
    set des_raw_data_path=!des_path!\raw_data
    if not exist !des_path! mkdir !des_path!
    if not exist !des_data_path! mkdir !des_data_path!
    if not exist !des_raw_data_path! mkdir !des_raw_data_path!
    copy !ref_path!\label_1.csv !des_data_path!\ref_label.csv
    copy !query_path!\label_1.csv !des_data_path!\query_label.csv
    copy !ref_path!\data_1.csv !des_raw_data_path!\ref_data.csv
    copy !ref_path!\label_1.csv !des_raw_data_path!\ref_label.csv
    copy !query_path!\data_1.csv !des_raw_data_path!\query_data.csv
    copy !query_path!\label_1.csv !des_raw_data_path!\query_label.csv
)
echo File copy successfully
echo start preprocess
auto.bat

echo start train
python main.py