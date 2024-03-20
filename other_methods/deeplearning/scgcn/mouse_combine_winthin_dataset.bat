@echo off
setlocal enabledelayedexpansion
set list[0]=10x_v3
set list[1]=cel_seq
set list[2]=dropseq
set list[3]=emtab5061
set list[4]=gse81608
set list[5]=gse84133_human
set list[6]=gse84133_mouse
set list[7]=gse85241
set list[8]=indrop
set list[9]=seq_well
set list[10]=smart_seq
set list[11]=mouse_combine

set rgraph=False
set basepath=E:\YuAnHuang\kevislin\Cell_Classification\experiment\within_dataset
for /l %%n in (0,1,10) do (
    Rscript data_preprocess2.R %basepath%\!list[%%n]! %rgraph%
    python train_for_compare.py --proj !list[%%n]!
)

@REM Speices
set basepath=E:\YuAnHuang\kevislin\Cell_Classification\experiment\species_v3
Rscript data_preprocess.R %basepath%\mouse_combine %rgraph%
python train_for_compare.py --proj mouse_combine
