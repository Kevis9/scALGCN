@echo off
setlocal enabledelayedexpansion
set list[0]=cel_seq_smart_seq
set list[1]=cel_seq_drop_seq
set list[2]=cel_seq_10x
set list[3]=seq_well_smart_seq
set list[4]=seq_well_drop_seq
set list[5]=seq_well_10x
set list[6]=indrop_drop_seq
set list[7]=indrop_10x
set list[8]=drop_seq_10x
set list[9]=drop_seq_smart_seq
set list[10]=84133_5061
set list[11]=84133_85241
set list[12]=84133_81608
set list[13]=gse\mouse_human
set list[14]=gse\human_mouse
set list[15]=gse_emtab\mouse_human
set list[16]=gse_emtab\human_mouse
set list[17]=mca_gse84133
set list[18]=cel_seq_10x_v3
set list[19]=seq_well_10x_v3
set list[20]=indrop_10x_v3
set list[21]=indrop_smart_seq
set list[22]=drop_seq_indrop
set list[23]=drop_seq_10x_v3
set list[24]=10x_v3_cel_seq
set list[25]=10x_v3_drop_seq
set list[26]=10x_v3_indrop
set list[27]=10x_v3_seq_well
set list[28]=smart_seq_seq_well
set list[29]=smart_seq_indrop
set list[30]=smart_seq_drop_seq
set list[31]=smart_seq_10x_v3
set list[32]=5061_84133
set list[33]=84133_combine
set list[34]=combine_84133
set list[35]=mouse_combine
set list[36]=combine_mouse
set list[37]=gsemouse_gse85241
set list[38]=gsemouse_emtab_leftjoin
set list[39]=lung
set list[40]=kidney
set list[41]=Cao_2020_stomach
set list[42]=GSE72056
set list[43]=GSE98638
set list[44]=GSE99254
set list[45]=GSE108989
set list[46]=GSE115746
set list[47]=GSM3271044
set list[48]=MacParland

set list[50]=Guo
set list[51]=He_Calvarial_Bone
set list[52]=Enge
set list[53]=Hu
set list[54]=Wu_human
set list[55]=Guo_2021
set list[56]=Loo_E14.5

set list[57]=GSE72056_GSE103322
set list[58]=GSE118056_GSE117988

set list[59]=Haber


set list[60]=gse\mouse_human
set list[61]=gse\human_mouse
set list[62]=mouse_combine
set list[63]=combine_mouse
set list[64]=cel_seq_drop_seq

set list[65]=gsemouse_gsehuman
set list[66]=gsehuman_gsemouse
set list[67]=mouse_combine
set list[68]=combine_mouse

set list[69]=gse\mouse_human
set list[70]=combine_mouse


set list[71]=GSE72056_GSE103322_B_cell
set list[72]=GSE72056_GSE103322_Endothelial
set list[73]=GSE72056_GSE103322_Macrophage
set list[74]=GSE72056_GSE103322_malignant
set list[75]=GSE72056_GSE103322_T_cell

set list[76]=GSE84133_EMTAB5061_alpha
set list[77]=GSE84133_EMTAB5061_beta
set list[78]=GSE84133_EMTAB5061_delta
set list[79]=GSE84133_EMTAB5061_gamma

set list[80]=GSE103322_GSE72056_malignant2
set list[81]=GSE72056_GSE103322_malignant

set list[82]=cel_seq_smart_seq
set list[83]=cel_seq_10x_v3
set list[84]=seq_well_smart_seq
set list[85]=seq_well_drop_seq
set list[86]=seq_well_10x_v3
set list[87]=smart_seq_10x_v3
set list[88]=indrop_drop_seq
set list[89]=indrop_10x_v3
set list[90]=indrop_smart_seq
set list[91]=drop_seq_smart_seq
set list[92]=drop_seq_10x_v3

set list[93]=gse\mouse_human
set list[94]=gse\human_mouse
set list[95]=mouse_combine
set list[96]=combine_mouse
set list[97]=cel_seq_smart_seq
set list[98]=cel_seq_10x_v3
set list[99]=seq_well_smart_seq
set list[100]=seq_well_drop_seq
set list[101]=seq_well_10x_v3
set list[102]=smart_seq_10x_v3
set list[103]=indrop_drop_seq
set list[104]=indrop_10x_v3
set list[105]=indrop_smart_seq
set list[106]=drop_seq_smart_seq
set list[107]=drop_seq_10x_v3

set list[108]=GSE96583

set rgraph=False
set basepath=E:\YuAnHuang\kevislin\Cell_Classification\experiment\perturbation
for /l %%n in (108,1,108) do (
    Rscript data_preprocess.R %basepath%\!list[%%n]! %rgraph%
    python train_for_compare.py --proj !list[%%n]!
@REM     python train_for_compare.py --proj !list[%%n]! --graph %rgraph%
)


