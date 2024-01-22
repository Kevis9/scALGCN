import pandas as pd
import random
import os
import argparse
root_path = 'D:\\YuAnHuang\\kevislin\\cancerSEA'
old_save_path = 'D:\\YuAnHuang\\kevislin\\scALGCN\\experiments'
parser = argparse.ArgumentParser()

ref_projs = ['EXP0004', 'EXP0061','EXP0063', 'EXP0050', 'EXP0059']
query_projs = ['EXP0004', 'EXP0061','EXP0063', 'EXP0050', 'EXP0059']

for ref_proj in ref_projs:
    for query_proj in query_projs:
        if ref_proj == query_proj:
            continue
        ref_data = pd.read_csv(os.path.join(root_path, ref_proj + '_PCG_log2_afterQC.txt'), delimiter='\t', index_col=0)
        query_data = pd.read_csv(os.path.join(root_path, query_proj + '_PCG_log2_afterQC.txt'), delimiter='\t', index_col=0)
        ref_data = ref_data.copy()
        ref_data = ref_data.drop(index=['Sample'])
        ref_data = ref_data.T
        
        query_data = query_data.copy()
        query_data = query_data.drop(index=['Sample'])
        query_data = query_data.T

        ref_label = pd.read_csv(os.path.join(root_path, ref_proj + '_Score.txt'), delimiter='\t', index_col=3)
        query_label = pd.read_csv(os.path.join(root_path, query_proj + '_Score.txt'), delimiter='\t', index_col=3)

        ref_label = ref_label.copy()
        ref_label = ref_label.drop(columns=['ExpID', 'cancer', 'sample'])

        query_label = query_label.copy()
        query_label = query_label.drop(columns=['ExpID', 'cancer', 'sample'])

        # gene intersection
        genes = set(ref_data.columns.tolist()) & set(query_data.columns.tolist())
        genes = list(genes)
        ref_data = ref_data.loc[:, genes]
        query_data = query_data.loc[:, genes]

        # save data and label
        save_path = os.path.join(old_save_path, ref_proj+'_'+query_proj)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.mkdir(os.path.join(save_path, 'raw_data'))
            os.mkdir(os.path.join(save_path, 'data'))
        
        ref_data.to_csv(os.path.join(save_path, 'raw_data', 'ref_data.csv'), index=True)
        query_data.to_csv(os.path.join(save_path, 'raw_data', 'query_data.csv'), index=True)

        ref_label.to_csv(os.path.join(save_path, 'raw_data', 'ref_label.csv'), index=True)
        query_label.to_csv(os.path.join(save_path, 'raw_data', 'query_label.csv'), index=True)
        ref_label.to_csv(os.path.join(save_path, 'data', 'ref_label.csv'), index=True)
        query_label.to_csv(os.path.join(save_path, 'data', 'query_label.csv'), index=True)

        print(ref_data.shape)
        print(query_data.shape)
        print(ref_label.shape)
        print(query_label.shape)