'''
    process data from single-cell portral 
    1. mtx expression file (log normalized data)
    2. gene name file
    3. cell name
    4. cell type
    into a h5ad file
'''
import pandas as pd
import scanpy as sc
import anndata as ad
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dir_name', type='str', 
                    default='')

args = parser.parse_args()
dir_name = args.dir_name

csr_data = sc.read(os.path.join(dir_name, 'data.mtx'))
gene_names = pd.read_csv(os.path.join(dir_name, 'gene.txt'), delimiter='\t', header=None)
meta_data = pd.read_csv(os.path.join(dir_name, 'meta_data.txt'), delimiter='\t')
'''
    !!! different meta data have differet columns, you should change the name keys here
'''
cell_names = meta_data['NAME'].tolist()
cell_types = meta_data['Cell_Type'].tolist()

adata = ad.AnnData(csr_data, dtype=float)
adata.obs_names = cell_names
adata.var_names = gene_names.iloc[:, 0].tolist()
adata.uns['cell_type'] = cell_types

adata.write(os.path.join(dir_name, 'data.h5ad'))


