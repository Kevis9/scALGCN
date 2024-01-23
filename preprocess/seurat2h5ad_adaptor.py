import pandas as pd
import anndata as ad
import argparse
import os
from scipy.sparse import csr_matrix
from scipy.io import mmread
import re
'''
    Process data from R, which produces:
    1. afterNorm_ref_data_middle.mtx: 
    2. afterNorm_query_data_middle.mtx: 
    3. afterNorm_auxilary_data_middle.mtx
    read them and turn into h5ad file
'''
def copy_h5ad(adata, data, genes, auxilary):
    new_data = ad.AnnData(data)
    new_data.var_names = genes
    new_data.obs_names = adata.obs_names
    if auxilary:
        new_data.obsm['label'] = adata.obsm['label']
    else:
        new_data.obs['cell_type'] = adata.obs['cell_type']
    return new_data
    

parser = argparse.ArgumentParser()
parser.add_argument('--dir_name', type=str, default='')

args = parser.parse_args()
dir_name = args.dir_name

norm_ref_data = mmread(os.path.join(dir_name, 'data', 'afterNorm_ref_data_middle.mtx')).tocsr()
norm_query_data = mmread(os.path.join(dir_name, 'data', 'afterNorm_query_data_middle.mtx')).tocsr()
norm_auxilary_data = mmread(os.path.join(dir_name, 'data', 'afterNorm_auxilary_data_middle.mtx')).tocsr()

ref_data = ad.read(os.path.join(dir_name, 'raw_data', 'ref_data.h5ad'))
query_data = ad.read(os.path.join(dir_name, 'raw_data', 'query_data.h5ad'))
auxilary_data = ad.read(os.path.join(dir_name, 'raw_data', 'auxilary_data.h5ad'))

genes_df = pd.read_csv(os.path.join(dir_name, 'data', 'selected_genes_middle.csv'), index_col=0)
genes = genes_df.iloc[:, 0].tolist()

new_ref_data = copy_h5ad(ref_data, norm_ref_data.transpose(), genes, False)
new_query_data = copy_h5ad(query_data, norm_query_data.transpose(), genes, False)
new_auxilary_data = copy_h5ad(auxilary_data, norm_auxilary_data.transpose(), genes, True)


new_ref_data.write(os.path.join(dir_name, 'data', 'ref_data.h5ad'))
new_query_data.write(os.path.join(dir_name, 'data', 'query_data.h5ad'))
new_auxilary_data.write(os.path.join(dir_name, 'data', 'auxilary_data.h5ad'))

# delete middle files
def delete_files_with_pattern(root_folder, pattern):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            if re.search(pattern, filename):
                print(f"Deleting: {file_path}")
                os.remove(file_path)

# 指定要递归删除文件的根目录和匹配的正则表达式
root_directory = args.dir_name
file_pattern = "middle"

# 调用函数删除文件
delete_files_with_pattern(root_directory, file_pattern)