@echo off
setlocal enabledelayedexpansion

set projects=10x_v3_drop_seq,^
10x_v3_indrop,^
5061_84133,^
84133_5061,^
84133_81608,^
84133_85241,^
84133_combine,^
drop_seq_10x_v3,^
drop_seq_seq_well,^
indrop_10x_v3,^
indrop_drop_seq,^
seq_well_10x_v3,^
seq_well_drop_seq

for %%i in (%projects%) do (
@REM     echo %%i
    Rscript preprocess.R E:\YuAnHuang\kevislin\second_proj\scALGCN\experiment\%%i
)
echo File copy successfully


