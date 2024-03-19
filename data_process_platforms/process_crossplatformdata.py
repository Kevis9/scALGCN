import pandas as pd
from scipy.io import mmread
import anndata as ad
import os
import argparse
import numpy as np
import random
from scipy.sparse import csr_matrix


dir_name = '/home/hwl/raw_data/crossplatform'
projs = ['10xv3','celseq','dropseq','indrop','seqwell','smartseq']

for proj in projs:
    data = pd.read_csv(os.path.join(dir_name, proj, 'data.csv'), index_col=0)
    label = pd.read_csv(os.path.join(dir_name, proj, 'label.csv'))    
    new_columns = data.columns.str.split('_').str[1]
    data.columns = new_columns
    gene_names = data.columns.tolist()
    cell_names = data.index.tolist()    
    cell_types = label.iloc[:, 0].tolist()
    adata = ad.AnnData(csr_matrix(data.to_numpy()), dtype=float)
    adata.obs_names = cell_names
    adata.var_names = gene_names
    adata.obs['cell_type'] = cell_types
    print(adata.shape)
    print(adata)
    adata.write(os.path.join(dir_name, proj, 'data.h5ad'))