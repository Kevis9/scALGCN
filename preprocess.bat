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
set projects=cel_seq_10x_v3


for %%i in (%projects%) do (
    python preprocess\h5ad2seurat_adaptor.py --dir_name=E:\YuAnHuang\kevislin\second_proj\scALGCN\experiments\%%i
    Rscript preprocess\preprocess.R E:\YuAnHuang\kevislin\second_proj\scALGCN\experiments\%%i
    python preprocess\seurat2h5ad_adaptor.py --dir_name=E:\YuAnHuang\kevislin\second_proj\scALGCN\experiments\%%i    
    @REM delete middle files
    set "directory=E:\YuAnHuang\kevislin\second_proj\scALGCN\experiments\%%i"
    set "pattern=middle"

    for /r "%directory%" %%f in (*%pattern%*) do (
        echo Deleting: %%f
        @REM del "%%f"
    )
    
    @REM copy D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\raw_data\ref_label.csv D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\data\ref_label.csv   
    @REM copy D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\raw_data\query_label.csv D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\data\query_label.csv
    @REM copy D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\raw_data\auxilary_label.csv D:\YuAnHuang\kevislin\scALGCN\experiments\%%i\data\auxilary_label.csv

)





