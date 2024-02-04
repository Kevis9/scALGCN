import pandas as pd
from scipy.io import mmread
import anndata as ad
import os
import argparse
import numpy as np
import random
from scipy.sparse import csr_matrix

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

parser.add_argument('--dir_name', type=str, 
                    default='D:/YuAnHuang/kevislin/cancerCellType/lep')
parser.add_argument('--save_path', type=str, 
                    default='D:/YuAnHuang/kevislin/scALGCN/experiments/lep5000_lep5000')

args = parser.parse_args()
dir_name = args.dir_name
save_path = args.save_path

data = pd.read_csv(os.path.join(dir_name, 'Table_S11.txt'), index_col=0, sep='\t')
meta_data = pd.read_csv(os.path.join(dir_name, 'MetaDataFromTableS11.csv')) # 手动去掉了第二行

gene_names = data.index.tolist()
cell_names = data.columns.tolist() 

cell_types = meta_data['biosample_id'].tolist()

data = data.T
adata = ad.AnnData(csr_matrix(data.to_numpy()), dtype=float)
adata.obs_names = cell_names
adata.var_names = gene_names
adata.obs['cell_type'] = cell_types

ref_adata, ref_idxs = random_stratify(np.array(cell_types), adata=adata, size=(5000/adata.n_obs))
query_adata, query_idxs = random_stratify(np.array(cell_types), adata=adata, size=(5000/adata.n_obs), selected_idx=ref_idxs)

print("ref idxs and query idxs intersections:")
print(set(ref_idxs) & set(query_idxs))

print(ref_adata.shape)
print(query_adata.shape)

ref_adata.write(os.path.join(save_path, 'ref_data.h5ad'))
query_adata.write(os.path.join(save_path, 'query_data.h5ad'))