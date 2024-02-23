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
set projects=bcp1_bcp2_exp0013,^
second_proj




for %%i in (%projects%) do (
    python preprocess\h5ad2seurat_adaptor.py --dir_name=experiments\%%i
    Rscript preprocess\preprocess.R experiments\%%i
    python preprocess\seurat2h5ad_adaptor.py --dir_name=experiments\%%i    
    
)





