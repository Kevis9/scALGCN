import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import pandas as pd
import anndata as ad
import numpy as np
import os


def run_umap(data):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    return embedding

def plot_umap(data, label, name):
    data_df = pd.DataFrame(data, columns=['x', 'y'])
    data_df['label'] = label
    sns.scatterplot(data=data_df, x='x', y='y', hue='label', s=12, linewidth=0)
    plt.savefig(name, dpi=300, transparent=True)    


def read_data(data_paths):
    datas = []
    for path in data_paths:
        data = ad.read_h5ad(path)
        datas.append(data)
    return datas

result_paths = [
    '/home/Users/kevislin/scALGCN/result/bcp1_6000-mp1_6000-exp0040_GT + AL',
    '/home/Users/kevislin/scALGCN/other_methods/r_methods/result/bcp1_6000-mp1_6000/seurat',    
]

methods = [
    'scALGT',
    'seurat',    
]

for i, path in enumerate(result_paths):
    if i == 0:
        ref_emb = np.load(os.path.join(path, 'ref_embeddings.npy'))
        query_emb = np.load(os.path.join(path, 'query_embeddings.npy'))
        query_pred = pd.read_csv(os.path.join(path, 'query_pred.csv')).iloc[:, 0].tolist()
        all_true = pd.read_csv(os.path.join(path, 'query_true.csv')).iloc[:, 0].tolist()
        n_ref = len(all_true) - len(query_pred)
        all_pred = all_true[:n_ref] + query_pred
        all_emb = np.concatenate([ref_emb, query_emb], axis=0)
        all_emb_2d = run_umap(all_emb)
        
    else:
        all_emb_2d = pd.read_csv(os.path.join(path, 'embeddings_2d.csv')) 
        all_pred = pd.read_csv(os.path.join(path, 'all_preds.csv')).iloc[:, 0].tolist()
    

    plot_umap(all_emb_2d, all_pred, methods[i]+'_'+'umap_pred')
    

