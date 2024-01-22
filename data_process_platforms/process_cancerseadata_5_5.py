import pandas as pd
import random
import os
root_path = 'D:\\YuAnHuang\\kevislin\\cancerSEA'
save_path = 'D:\\YuAnHuang\\kevislin\\scALGCN\\experiments'

projs = ['EXP0001']
for proj in projs:
    pcg_data = pd.read_csv(os.path.join(root_path, proj + '_PCG_log2_afterQC.txt'), delimiter='\t', index_col=0)
    pcg_data = pcg_data.copy()
    pcg_data = pcg_data.drop(index=['Sample'])
    pcg_data = pcg_data.T

    cell_state = pd.read_csv(os.path.join(root_path, proj + '_Score.txt'), delimiter='\t', index_col=3)
    cell_state = cell_state.copy()
    cell_state = cell_state.drop(columns=['ExpID', 'cancer', 'sample'])

    n = pcg_data.shape[0]
    idxs = [i for i in range(n)]

    random.shuffle(idxs)

    mid = int(0.5 * n)
    ref_data = pcg_data.iloc[idxs[:mid], :]
    query_data = pcg_data.iloc[idxs[mid:], :]

    ref_label = cell_state.iloc[idxs[:mid], :]
    query_label = cell_state.iloc[idxs[mid:], :]

    # save data and label
    save_path = os.path.join(save_path, proj)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'raw_data'))
        os.mkdir(os.path.join(save_path, 'data'))
    
    ref_data.to_csv(os.path.join(save_path, 'raw_data', 'ref_data.csv'), index=True)
    query_data.to_csv(os.path.join(save_path, 'raw_data', 'query_data.csv'), index=True)

    ref_label.to_csv(os.path.join(save_path, 'raw_data', 'ref_label.csv'), index=True)
    query_label.to_csv(os.path.join(save_path, 'raw_data', 'query_label.csv'), index=True)

    print(ref_data.shape)
    print(query_data.shape)
    print(ref_label.shape)
    print(query_label.shape)

