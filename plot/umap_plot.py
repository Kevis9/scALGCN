import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import pandas as pd
import anndata as ad

def run_umap(data):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    return embedding

def plot_umap(data, label):
    data_df = pd.DataFrame(data, columns=['x', 'y'])
    data_df['label'] = label
    sns.scatterplot(data=data_df, x='x', y='y', hue='label', s=8, linewidth=0)
    plt.savefig('bcp1-bcp2-bcp3-raw', dpi=300, transparent=True)    


def read_data(data_paths):
    datas = []
    for path in data_paths:
        data = ad.read_h5ad(path)
        datas.append(data)
    return datas

data_paths = [
    '/home/Users/kevislin/raw_data/Wu2021/BC-P1_6000/data.h5ad',
    '/home/Users/kevislin/raw_data/Wu2021/BC-P1_6000/data.h5ad',
    '/home/Users/kevislin/raw_data/Wu2021/BC-P3_6000/data.h5ad'    
    '/home/Users/kevislin/raw_data/Wu2021/PC-P1/data.h5ad'    
]

datas = read_data(data_paths)

all_data = np.concatenate([data.X.toarray() for data in datas], axis=0)
print(all_data.shape)
bcp1_label = ['BC-P1' for i in range(datas[0].n_obs)]
bcp2_label = ['BC-P2' for i in range(datas[1].n_obs)]
bcp3_label = ['BC-P3' for i in range(datas[2].n_obs)]
pcp1_label = ['PC-P1' for i in range(datas[3].n_obs)]
all_label = bcp1_label + bcp2_label + bcp3_label + pcp1_label

embeddings = run_umap(all_data)
np.save('bcp1-bcp2-bcp3-raw-embeddings.npy', embeddings)
plot_umap(embeddings, all_label)

