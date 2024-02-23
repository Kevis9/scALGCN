import pandas as pd
import anndata as ad
import argparse
import os
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
import numpy as np
'''
    1. extract sparse matrix from data.h5ad and make them appliacable for R    
    2. make intersection on genes and cell types
'''

parser = argparse.ArgumentParser()
parser.add_argument('--dir_name', type=str, default='')

args = parser.parse_args()
dir_name = args.dir_name
ref_data_h5 = ad.read_h5ad(os.path.join(dir_name, 'raw_data', 'ref_data.h5ad'))
query_data_h5 = ad.read_h5ad(os.path.join(dir_name, 'raw_data', 'query_data.h5ad'))
auxilary_data_h5 = ad.read_h5ad(os.path.join(dir_name, 'raw_data', 'auxilary_data.h5ad'))

ref_obs_names = ref_data_h5.obs_names.to_numpy()
query_obs_names = query_data_h5.obs_names.to_numpy()

# First ref and query do cell intersection
ref_data = ref_data_h5.X
query_data = query_data_h5.X
auxilary_data = auxilary_data_h5.X
ref_cell_types = np.array(ref_data_h5.obs['cell_type'].tolist())
query_cell_types = np.array(ref_data_h5.obs['cell_type'].tolist())
common_types = list(set(ref_cell_types) & set(query_cell_types))
ref_idx = np.where(np.isin(ref_cell_types, common_types))[0]
query_idx = np.where(np.isin(query_cell_types, common_types))[0]

ref_label = ref_cell_types[ref_idx]
query_label = query_cell_types[query_idx]
ref_obs_names = ref_obs_names[ref_idx]
query_obs_names = query_obs_names[query_idx]

ref_data = ref_data[ref_idx, :]
query_data = query_data[query_idx, :]


# secode: gene intersection, we don't need to change the data here, since the slice operation will happen in preprocess.R
common_gene = list(set(ref_data_h5.var_names.tolist()) & set(query_data_h5.var_names.tolist()) & set(auxilary_data_h5.var_names.tolist()))
common_gene_df = pd.DataFrame(common_gene, columns=['gene_name'])
common_gene_df.to_csv(os.path.join(dir_name, 'raw_data', 'common_gene_middle.csv'), index=False)


ref_label = pd.DataFrame(ref_label, columns=['cell_type'])
ref_label.to_csv(os.path.join(dir_name, 'raw_data', 'ref_label_middle.csv'), index=False)

query_label = pd.DataFrame(query_label, columns=['cell_type'])
query_label.to_csv(os.path.join(dir_name, 'raw_data', 'query_label_middle.csv'), index=False)

ref_obs_names = pd.DataFrame(ref_obs_names, columns=['cell_name'])
ref_obs_names.to_csv(os.path.join(dir_name, 'raw_data', 'ref_name_middle.csv'), index=False)

query_obs_names = pd.DataFrame(query_obs_names, columns=['cell_name'])
query_obs_names.to_csv(os.path.join(dir_name, 'raw_data', 'query_name_middle.csv'), index=False)


mmwrite(os.path.join(dir_name, 'raw_data', 'ref_data_middle.mtx'), ref_data)
mmwrite(os.path.join(dir_name, 'raw_data', 'query_data_middle.mtx'), query_data)
mmwrite(os.path.join(dir_name, 'raw_data', 'auxilary_data_middle.mtx'), auxilary_data)



