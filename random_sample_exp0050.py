import anndata as ad

sample_num = ['200', '400', '600', '800', '1000']

adata = ad.read_h5ad('/home/hwl/raw_data/cell_states_data/EXP0050/auxilary_data.h5ad')

for num in sample_num:    
    obs_indices = adata.obs_names[:num]    
    sampled_adata = adata[obs_indices, :]
    sampled_adata.write_h5ad(f'/home/hwl/raw_data/cell_states_data/EXP0050/auxilary_data_{num}.h5ad')
