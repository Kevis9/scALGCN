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
import numpy as np
import random

def random_stratify(cell_types, adata, size, selected_idx=[]):
    classes = np.unique(cell_types)
    all_select_idxs = []
    for t in list(classes):
        idx = list(np.where(cell_types==t)[0])
        # delete seleted idxs
        idx = list(set(idx) - set(selected_idx))
        random.shuffle(idx)        
        s_idx = idx[:int(size * len(idx))]
        all_select_idxs += s_idx
    new_adata = adata[all_select_idxs, :]
    return new_adata, all_select_idxs
        
    


parser = argparse.ArgumentParser()

parser.add_argument('--dir_name', type='str', 
                    default='')
parser.add_argument('--save_path', type='str', 
                    default='')

args = parser.parse_args()
dir_name = args.dir_name
save_path = args.save_path

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
adata.obs['cell_type'] = cell_types

ref_adata, ref_idxs = random_stratify(np.array(cell_types), adata=adata, size=(5000/adata.n_obs))
query_adata, query_idxs = random_stratify(np.array(cell_types), adata=adata, size=(5000/adata.n_obs), selected_idx=ref_idxs)

print("ref idxs and query idxs intersections:")
print(set(ref_idxs) & set(query_idxs))

ref_adata.write(os.path.join(save_path, 'ref_data.h5ad'))
query_adata.write(os.path.join(save_path, 'query_data.h5ad'))
