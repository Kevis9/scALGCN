import pandas as pd
import anndata as ad
import argparse
import os
from scipy.sparse import csr_matrix
from scipy.io import mmwrite

'''
    extract sparse matrxi from data.h5ad and make them appliacable for R    
'''

parser = argparse.ArgumentParser()
parser.add_argument('--dir_name', type=str, default='')

args = parser.parse_args()
dir_name = args.dir_name
ref_data_h5 = ad.read(os.path.join(dir_name, 'raw_data', 'ref_data.h5ad'))
query_data_h5 = ad.read(os.path.join(dir_name, 'raw_data', 'query_data.h5ad'))
auxilary_data_h5 = ad.read(os.path.join(dir_name, 'raw_data', 'auxilary_data.h5ad'))

ref_gene = pd.DataFrame(ref_data_h5.var_names, columns=['gene_name'])
query_gene = pd.DataFrame(query_data_h5.var_names, columns=['gene_name'])
auxilary_gene = pd.DataFrame(auxilary_data_h5.var_names, columns=['gene_name'])

ref_gene.to_csv(os.path.join(dir_name, 'raw_data', 'ref_gene_middle.csv'), index=False)
query_gene.to_csv(os.path.join(dir_name, 'raw_data', 'query_gene_middle.csv'), index=False)
auxilary_gene.to_csv(os.path.join(dir_name, 'raw_data', 'auxilary_gene_middle.csv'), index=False)

ref_label = pd.DataFrame(ref_data_h5.obs['cell_type'], columns=['cell_type'])
ref_label.to_csv(os.path.join(dir_name, 'raw_data', 'ref_label_middle.csv'), index=False)

ref_data = ref_data_h5.X
query_data = query_data_h5.X
auxilary_data = auxilary_data_h5.X

mmwrite(os.path.join(dir_name, 'raw_data', 'ref_data_middle.mtx'), ref_data)
mmwrite(os.path.join(dir_name, 'raw_data', 'query_data_middle.mtx'), query_data)
mmwrite(os.path.join(dir_name, 'raw_data', 'auxilary_data_middle.mtx'), auxilary_data)



