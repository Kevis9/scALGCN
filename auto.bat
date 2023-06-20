@echo off
setlocal enabledelayedexpansion
set list[0]=baron_xin
for /l %%n in (0,1,0) do (
    Rscript preprocess.R E:\YuAnHuang\kevislin\Cell_Classification\experiment\!list[%%n]!
)
