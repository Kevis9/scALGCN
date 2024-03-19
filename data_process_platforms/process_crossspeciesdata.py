import pandas as pd
from scipy.io import mmread
import anndata as ad
import os
import argparse
import numpy as np
import random
from scipy.sparse import csr_matrix


dir_name = '/home/hwl/raw_data/crossspecies'
projs = ['GSE84133/raw_data_with_same_gene_from_singlecellNet/mouse','GSE84133/raw_data_with_same_gene_from_singlecellNet/human', 'mouse_combine_from_graphcs/human']

for proj in projs:
    data = pd.read_csv(os.path.join(dir_name, proj, 'data.csv'), index_col=0)
    label = pd.read_csv(os.path.join(dir_name, proj, 'label.csv'))
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