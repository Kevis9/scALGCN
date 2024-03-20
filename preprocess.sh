#!/bin/bash

# projects="bcp1_bcp2_exp0013,\
# second_proj"


projects=(
    "celseq_10xv3" \
    "celseq_dropseq" \
    "celseq_indrop" \
    "celseq_seqwell" \
    "celseq_smartseq" \
    "10xv3_celseq" \
    "10xv3_dropseq" \
    "10xv3_indrop" \
    "10xv3_seqwell" \
    "10xv3_smartseq" \
    "dropseq_celseq" \
    "dropseq_10xv3" \
    "dropseq_indrop" \
    "dropseq_seqwell" \
    "dropseq_smartseq" \
    "indrop_celseq" \
    "indrop_10xv3" \
    "indrop_dropseq" \
    "indrop_seqwell" \
    "indrop_smartseq" \
    "seqwell_celseq" \
    "seqwell_10xv3" \
    "seqwell_dropseq" \
    "seqwell_indrop" \
    "seqwell_smartseq" \
    "smartseq_celseq" \
    "smartseq_10xv3" \
    "smartseq_dropseq" \
    "smartseq_indrop" \
    "smartseq_seqwell" \
    "mouse_human" \
    "human_mouse" \
    "mouse_humancombine" \
    "humancombine_mouse"
)

for i in ${projects[@]}; do        
    python preprocess/h5ad2seurat_adaptor.py --dir_name=experiments/$i
    Rscript preprocess/preprocess.R experiments/$i
    python preprocess/seurat2h5ad_adaptor.py --dir_name=experiments/$i
done