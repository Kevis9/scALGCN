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
ref_label.to_csv('ref_label_middle.csv', index=False)

ref_data_h5.obs.drop(ref_data_h5.obs.index, inplace=True)
query_data_h5.obs.drop(query_data_h5.obs.index, inplace=True)
auxilary_data_h5.obs.drop(auxilary_data_h5.obs.index, inplace=True)

ref_data_h5.write(os.path.join(dir_name, 'raw_data', 'ref_data_middle.h5ad'))
query_data_h5.write(os.path.join(dir_name, 'raw_data', 'query_data_middle.h5ad'))
auxilary_data_h5.write(os.path.join(dir_name, 'raw_data', 'auxilary_data_middle.h5ad'))





