import pandas as pd
import random
import os
import anndata as ad
from scipy.sparse import csr_matrix
root_path = '~/raw_data/cell_states_data'
save_path = '~/raw_data/cell_states_data'
'''
pipeline:
    1. modify the expression data to cell * gene (index and columns only contains cell names and gene names)
    2. change gene IDs to gene names
'''

projs = ['EXP0047', 'EXP0013', 'EXP0050']


for i, proj in enumerate(projs):
    pcg_data = pd.read_csv(os.path.join(root_path, proj, proj + '_PCG_log2_afterQC.txt'), delimiter='\t', index_col=0)
    pcg_data = pcg_data.copy()
    pcg_data = pcg_data.drop(index=['Sample'])
    pcg_data = pcg_data.T

    cell_state = pd.read_csv(os.path.join(root_path, proj, proj + '_Score.txt'), delimiter='\t', index_col=3)
    cell_state = cell_state.copy()
    cell_state = cell_state.drop(columns=['ExpID', 'cancer', 'sample'])
        
    id_name_map = pd.read_csv('ensemble_id_gene_name.csv')
    id_name_map = dict(zip(id_name_map['ID'].tolist(), id_name_map['name'].tolist()))
    idx = pcg_data.columns.isin(id_name_map.keys()).tolist()
    pcg_data = pcg_data.iloc[:, idx]
    
    pcg_data.columns = pcg_data.columns.map(lambda x: id_name_map[x])    

    adata = ad.AnnData(csr_matrix(pcg_data.to_numpy(dtype=float)))
    adata.obs_names = pcg_data.index.tolist()
    adata.var_names = pcg_data.columns.tolist()
    adata.obsm['label'] = cell_state.to_numpy()
    print(adata)
    save_path = os.path.join(save_path, proj)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    adata.write(os.path.join(save_path, 'auxilary_data.h5ad'))
    print('save successfully!')