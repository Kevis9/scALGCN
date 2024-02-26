#!/bin/bash

# projects="bcp1_bcp2_exp0013,\
# second_proj"


projects=(    
          "bcp1_6000-bcp2_6000-exp0047" \
          "bcp1_6000-bcp3_6000-exp0013" \
          "bcp1_6000-bcp3_6000-exp0050" \
          "bcp1_6000-bcp3_6000-exp0047" \
          "bcp2_6000-bcp3_6000-exp0013" \
          "bcp2_6000-bcp3_6000-exp0047" \
          "bcp2_6000-bcp3_6000-exp0050" \
          "bcp1_6000-pcp1_6000-exp0013" \
          "bcp1_6000-pcp1_6000-exp0038" \
          "bcp1_6000-pcp1_6000-exp0050" \
          "bcp1_6000-pcp1_6000-exp0047" \
          "bcp1_6000-mp1_6000-exp0013" \
          "bcp1_6000-mp1_6000-exp0040" \
          "bcp1_6000-mp1_6000-exp0050" \
          "bcp1_6000-mp1_6000-exp0047")

reverse_projs=("bcp2_6000-bcp1_6000-exp0013" \
"bcp2_6000-bcp1_6000-exp0050" \
"bcp2_6000-bcp1_6000-exp0047" \
"bcp3_6000-bcp1_6000-exp0013" \
"bcp3_6000-bcp1_6000-exp0050" \
"bcp3_6000-bcp1_6000-exp0047" \
"bcp3_6000-bcp2_6000-exp0013" \
"bcp3_6000-bcp2_6000-exp0047" \
"bcp3_6000-bcp2_6000-exp0050" \
"pcp1_6000-bcp1_6000-exp0013" \
"pcp1_6000-bcp1_6000-exp0038" \
"pcp1_6000-bcp1_6000-exp0050" \
"pcp1_6000-bcp1_6000-exp0047" \
"mp1_6000-bcp1_6000-exp0013" \
"mp1_6000-bcp1_6000-exp0040" \
"mp1_6000-bcp1_6000-exp0050" \
"mp1_6000-bcp1_6000-exp0047")

for i in ${projects[@]}; do    
    python preprocess/h5ad2seurat_adaptor.py --dir_name=experiments/$i
    Rscript preprocess/preprocess.R experiments/$i
    python preprocess/seurat2h5ad_adaptor.py --dir_name=experiments/$i
done

for i in ${reverse_projs[@]}; do    
    python preprocess/h5ad2seurat_adaptor.py --dir_name=experiments/$i
    Rscript preprocess/preprocess.R experiments/$i
    python preprocess/seurat2h5ad_adaptor.py --dir_name=experiments/$i
done