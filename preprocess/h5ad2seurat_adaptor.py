import pandas as pd
import anndata as ad
import argparse
import os
'''
    Now the ref_data.h5ad contains X and obs['cell_type']
    In order to make SeuratDisk run successfully,
    we need to creat files as follow:
    1. h5ad file just containing X and obs_names and var_names (ref, query and auxilary)
    2. ref_label to csv file
'''

parser = argparse.ArgumentParser()
parser.add_argument('--dir_name', type=str, default='')

args = parser.parse_args()
dir_name = args.dir_name
ref_data_h5 = ad.read(os.path.join(dir_name, 'raw_data', 'ref_data.h5ad'))
query_data_h5 = ad.read(os.path.join(dir_name, 'raw_data', 'query_data.h5ad'))
auxilary_data_h5 = ad.read(os.path.join(dir_name, 'raw_data', 'auxilary_data.h5ad'))

ref_label = pd.DataFrame(ref_data_h5.obs['cell_type'], columns=['cell_type'])
ref_label.to_csv(os.path.join(dir_name, 'raw_data', 'ref_label_middle.csv'), index=False)

new_ref_data = ad.AnnData(ref_data_h5.X)
new_ref_data.obs_names = ref_data_h5.obs_names
new_ref_data.var_names = ref_data_h5.var_names

new_query_data = ad.AnnData(query_data_h5.X)
new_query_data.obs_names = query_data_h5.obs_names
new_query_data.var_names = query_data_h5.var_names

new_auxilary_data = ad.AnnData(auxilary_data_h5.X)
new_auxilary_data.obs_names = auxilary_data_h5.obs_names
new_auxilary_data.var_names = auxilary_data_h5.var_names

new_ref_data.write(os.path.join(dir_name, 'raw_data', 'ref_data_middle.h5ad'))
new_query_data.write(os.path.join(dir_name, 'raw_data', 'query_data_middle.h5ad'))
new_auxilary_data.write(os.path.join(dir_name, 'raw_data', 'auxilary_data_middle.h5ad'))





