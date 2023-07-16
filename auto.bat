@echo off
setlocal enabledelayedexpansion
set list[0]=baron_xin
set list[1]=cel_seq_drop_seq
for /l %%n in (1,1,1) do (
    Rscript preprocess.R E:\YuAnHuang\kevislin\second_proj\scALGCN\experiment\!list[%%n]!
)
