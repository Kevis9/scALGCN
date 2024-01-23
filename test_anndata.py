import pandas as pd
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
import os
proj = 'cel_seq_10x_v3'

counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
adata = ad.AnnData(counts)
# adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
# adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
adata.write('test_data.h5ad')
exit()

path = os.path.join('experiments', proj, 'raw_data')
ref_data = pd.read_csv(os.path.join(path, 'ref_data.csv'), index_col=0)
ref_label = pd.read_csv(os.path.join(path, 'ref_label.csv'))
query_data = pd.read_csv(os.path.join(path, 'query_data.csv'), index_col=0)
query_label = pd.read_csv(os.path.join(path, 'query_label.csv'))
auxilary_data = pd.read_csv(os.path.join(path, 'auxilary_data.csv'), index_col=0)
auxilary_label = pd.read_csv(os.path.join(path, 'auxilary_label.csv'), index_col=0)


ref_adata = ad.AnnData(csr_matrix(ref_data.to_numpy()))
ref_adata.obs_names = ref_data.index.tolist()
ref_adata.var_names = ref_data.columns.tolist()
ref_adata.obs['cell_type'] = ref_label.to_numpy().reshape(-1)

query_adata = ad.AnnData(csr_matrix(query_data.to_numpy()))
query_adata.obs_names = query_data.index.tolist()
query_adata.var_names = query_data.columns.tolist()
query_adata.obs['cell_type'] = query_label.to_numpy().reshape(-1)

auxilary_adata = ad.AnnData(csr_matrix(auxilary_data.to_numpy()))
auxilary_adata.obs_names = auxilary_data.index.tolist()
auxilary_adata.var_names = auxilary_data.columns.tolist()
auxilary_adata.obsm['label'] = auxilary_label.to_numpy()

ref_adata.write(os.path.join(path, 'ref_data.h5ad'))
query_adata.write(os.path.join(path, 'query_data.h5ad'))
auxilary_adata.write(os.path.join(path, 'auxilary_data.h5ad'))
