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
set projects=wu2021_5000_wu2021_5000_exp0047



for %%i in (%projects%) do (
    python preprocess\h5ad2seurat_adaptor.py --dir_name=D:\YuAnHuang\kevislin\scALGCN\experiments\%%i
    Rscript preprocess\preprocess.R D:\YuAnHuang\kevislin\scALGCN\experiments\%%i
    python preprocess\seurat2h5ad_adaptor.py --dir_name=D:\YuAnHuang\kevislin\scALGCN\experiments\%%i    
    
)





