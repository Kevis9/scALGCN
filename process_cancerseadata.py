import pandas as pd
import random
pcg_data = pd.read_csv('EXP0001_PCG_log2_afterQC.txt', delimiter='\t', index_col=0)
pcg_data = pcg_data.copy()
pcg_data = pcg_data.drop(index=['Sample'])
pcg_data = pcg_data.T

cell_state = pd.read_csv('EXP0001_Score.txt', delimiter='\t', index_col=3)
cell_state = cell_state.copy()
cell_state = cell_state.drop(columns=['ExpID', 'cancer', 'sample'])

n = len(pcg_data.shape[0])
idxs = [i for i in range(n)]

random.shuffle(idxs)

mid = int(0.5 * n)
ref_data = pcg_data.loc[idxs[:mid], :]
query_data = pcg_data.loc[idxs[mid:], :]

ref_label = cell_state.loc[idxs[:mid], :]
query_label = cell_state.loc[idxs[mid:], :]

# save data and label
ref_data.to_csv('ref_data.csv', index=True)
query_data.to_csv('query_data.csv', index=True)

ref_label.to_csv('ref_label.csv', index=True)
query_label.to_csv('query_label.csv', index=True)

print(ref_data.shape)
print(query_data.shape)
print(ref_label.shape)
print(query_label.shape)

