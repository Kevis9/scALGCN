#!/bin/bash

# projects="bcp1_bcp2_exp0013,\
# second_proj"


projects=(
    "mp1_6000-pcp1_6000-exp0050" \
    "mp1_6000-pcp1_6000" \
    "pcp1_6000-mp1_6000-exp0050" \
    "pcp1_6000-mp1_6000"    
)

for i in ${projects[@]}; do        
    python preprocess/h5ad2seurat_adaptor.py --dir_name=experiments/$i
    Rscript preprocess/preprocess.R experiments/$i FALSE
    python preprocess/seurat2h5ad_adaptor.py --dir_name=experiments/$i
done