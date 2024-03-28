import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import pandas as pd
import anndata as ad
import numpy as np
import os
from sklearn.metrics import silhouette_score, adjusted_rand_score

def run_umap(data):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    return embedding

def plot_umap(data, label, name):
    data_df = pd.DataFrame(data, columns=['x', 'y'])
    data_df['label'] = label
    sns.scatterplot(data=data_df, x='x', y='y', hue='label', s=12, linewidth=0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    plt.savefig(name, dpi=300, transparent=True,bbox_inches="tight")    
    plt.clf()

def read_data(data_paths):
    datas = []
    for path in data_paths:
        data = ad.read_h5ad(path)
        datas.append(data)
    return datas

result_paths = [
    '/home/hwl/scALGCN/result/bcp3_6000-bcp2_6000-exp0050_GT + GL',
    '/home/hwl/scALGCN/result/bcp3_6000-bcp2_6000_GT + GL',
]

methods = [
    'scALGT_with_aux',
    'scALGT_without_aux',
]

for i, path in enumerate(result_paths):
    
    ref_emb = np.load(os.path.join(path, 'ref_embeddings.npy'))
    query_emb = np.load(os.path.join(path, 'query_embeddings.npy'))
    query_pred = pd.read_csv(os.path.join(path, 'query_pred.csv')).iloc[:, 0].tolist()
    all_true = pd.read_csv(os.path.join(path, 'query_true.csv')).iloc[:, 0].tolist()
    n_ref = len(all_true) - len(query_pred)
    all_pred = all_true[:n_ref] + query_pred        
    all_emb = np.concatenate([ref_emb, query_emb], axis=0)
    all_emb_2d = run_umap(all_emb)    
    print(methods[i])
    print(all_emb_2d.shape)
    print("ARI, {:.3f}, sil score {:.3f}".format(adjusted_rand_score(all_true, all_pred), silhouette_score(all_emb_2d, all_pred)))
    plot_umap(all_emb_2d, all_pred, methods[i]+'_'+'umap_pred')
    

