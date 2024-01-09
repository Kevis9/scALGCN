@echo off
setlocal enabledelayedexpansion

set projects=EXP0001
@REM kidney,^
@REM lung

for %%i in (%projects%) do (
    Rscript preprocess.R D:\YuAnHuang\kevislin\scALGCN\experiments\%%i
)


