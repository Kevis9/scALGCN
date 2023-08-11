@echo off
setlocal enabledelayedexpansion

set projects=A549,^
kidney,^
lung

for %%i in (%projects%) do (
@REM     echo %%i
    Rscript preprocess.R E:\YuAnHuang\kevislin\second_proj\scALGCN\experiment\%%i
)


