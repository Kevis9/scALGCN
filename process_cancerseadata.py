import pandas as pd
import random
import os
root_path = 'D:\\YuAnHuang\\kevislin\\cancerSEA'
save_path = 'D:\\YuAnHuang\\kevislin\\cancerSEA\\processed_data'
'''
pipeline:
    1. modify the expression data to cell * gene (index and columns only contains cell names and gene names)
    2. change gene IDs to gene names    
'''
projs = ['EXP0001']
for proj in projs:
    pcg_data = pd.read_csv(os.path.join(root_path, proj + '_PCG_log2_afterQC.txt'), delimiter='\t', index_col=0)
    pcg_data = pcg_data.copy()
    pcg_data = pcg_data.drop(index=['Sample'])
    pcg_data = pcg_data.T

    cell_state = pd.read_csv(os.path.join(root_path, proj + '_Score.txt'), delimiter='\t', index_col=3)
    cell_state = cell_state.copy()
    cell_state = cell_state.drop(columns=['ExpID', 'cancer', 'sample'])
        
    id_name_map = pd.read_csv('ensemble_id_gene_name.csv')
    id_name_map = dict(zip(id_name_map['ID'].tolist(), id_name_map['name'].tolist()))
    idx = pcg_data.columns.isin(id_name_map.keys()).tolist()
    pcg_data = pcg_data.iloc[:, idx]
    
    pcg_data.columns = pcg_data.columns.map(lambda x: id_name_map[x])    
    
    print(pcg_data.columns)
    
    # save data and label
    if not os.path.exists(save_path):
        os.mkdir(save_path)    
    save_path = os.path.join(save_path, proj)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    pcg_data.to_csv(os.path.join(save_path, 'auxilary_data.csv'))
    cell_state.to_csv(os.path.join(save_path, 'auxilary_label.csv'))