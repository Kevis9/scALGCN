@echo off
setlocal enabledelayedexpansion

@REM set projects=EXP0004_EXP0061,^
@REM EXP0004_EXP0063,^
@REM EXP0004_EXP0050,^
@REM EXP0004_EXP0059,^
@REM EXP0061_EXP0004,^
@REM EXP0061_EXP0063,^
@REM EXP0061_EXP0050,^
@REM EXP0061_EXP0059,^
@REM EXP0063_EXP0004,^
@REM EXP0063_EXP0061,^
@REM EXP0063_EXP0050,^
@REM EXP0063_EXP0059,^
@REM EXP0050_EXP0059,^
@REM EXP0050_EXP0004,^
@REM EXP0050_EXP0061,^
@REM EXP0050_EXP0063,^
@REM EXP0059_EXP0004,^
@REM EXP0059_EXP0061,^
@REM EXP0059_EXP0063,^
@REM EXP0059_EXP0050
set projects=seq_well_10x_v3_EXP0001


for %%i in (%projects%) do (
    Rscript preprocess.R D:\YuAnHuang\kevislin\scALGCN\experiments\%%i
    copy D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\raw_data\ref_label.csv D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\data\ref_label.csv   
    copy D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\raw_data\query_label.csv D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\data\query_label.csv
    copy D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\raw_data\auxilary_label.csv D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\data\auxilary_label.csv

)


