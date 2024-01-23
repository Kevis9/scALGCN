import pandas as pd
import anndata as ad
import argparse
import os
'''
    Process data from R, which produces:
    1. afterNorm_ref_data_middle.h5ad: 
    2. afterNorm_query_data_middle.h5ad: 
    3. afterNorm_auxilary_data_middle.h5ad
    both only containing X and obs_names and var_nams (We only need norm X)
    need to combine with ref_data.h5ad and query_data.h5ad and auxilary_data.h5ad    
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dir_name', type=str, default='')

args = parser.parse_args()
dir_name = args.dir_name

norm_ref_data = ad.read(os.path.join(dir_name, 'data', 'afterNorm_ref_data_middle.h5ad'))
norm_query_data = ad.read(os.path.join(dir_name, 'data', 'afterNorm_query_data_middle.h5ad'))
norm_auxilary_data = ad.read(os.path.join(dir_name, 'data', 'afterNorm_auxilary_data_middle.h5ad'))

ref_data = ad.read(os.path.join(dir_name, 'raw_data', 'ref_data.h5ad'))
query_data = ad.read(os.path.join(dir_name, 'raw_data', 'query_data.h5ad'))
auxilary_data = ad.read(os.path.join(dir_name, 'raw_data', 'auxilary_data.h5ad'))

ref_data.X = norm_ref_data.X
query_data.X = norm_query_data.X
auxilary_data.X = norm_auxilary_data.X

ref_data.write(os.path.join(dir_name, 'data', 'ref_data.h5ad'))
query_data.write(os.path.join(dir_name, 'data', 'query_data.h5ad'))
auxilary_data.write(os.path.join(dir_name, 'data', 'auxilary_data.h5ad'))
