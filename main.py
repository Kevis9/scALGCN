import scipy.stats
import torch
import os
from utils import read_data_with_mm, capsule_pd_data_to_anndata, read_data_with_csv, combine_inter_intra_graph, centralissimo, perc_for_entropy, perc_for_density
import anndata as ad
import pandas as pd
import torch.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
from torch_geometric.utils import to_dense_adj
import numpy as np
# Data preparation

data_config = {
    'ref_data_path': 'ref_data',
    'query_data_path': 'query_data',
    'ref_label_path': 'ref_label',
    'query_label_path': 'query_label',
    'anndata_path': 'anndata.h5ad',
    'inter_graph_path': '',
    'intra_graph_path': '',
    'data_mode': 'csv'
}

parameter_config = {
    'gcn_hidden_units': 256,
    'epochs': 200,
    'gcn_lr': 1e-3,
    'basef': 0.99,
}

root_data_path = 'experiment/Baron_Xin'
for key in data_config:
    if "path" in key:
        data_config[key] = os.path.join(root_data_path, data_config[key])


def get_anndata():
    if data_config['data_mode'] == 'csv':
        ref_data = pd.read_csv(data_config['ref_data_path'], index_col=0)
        ref_label = pd.read_csv(data_config['ref_label_path'])
        query_data = pd.read_csv(data_config['query_data_path'], index_col=0)
        query_label = pd.read_csv(data_config['ref_label_path'])
        data = pd.concat([ref_data, query_data], axis=0)
        label = pd.concat([ref_label, query_label], axis=0)
        edge_index = combine_inter_intra_graph(inter_graph_path=data_config['inter_graph_path'],
                                               intra_graph_path=data_config['intra_graph_path'],
                                               n_nodes_ref=len(ref_label),
                                               n_nodes_query=len(query_label))
        adata = capsule_pd_data_to_anndata(data, label, edge_index)

        # 设置好train val test mask位置

        # adata.uns['train_mask'] = [i for i in range]
        # adata.uns['test_mask'] =
        # adata.uns['val_mask'] =
        adata.write(os.path.join(root_data_path, 'data.h5ad'), compression="gzip")
        return adata
    elif data_config['data_mode'] == 'ann':
        '''ann data已存在'''
        adata = ad.read_hdf(data_config['anndata_path'])
        return adata



def train(model, g_data):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=parameter_config['gcn_lr'],
                                 weight_decay=5e-4)

    dense_adj = to_dense_adj(g_data.edge_index).cpu().numpy()
    norm_centrality = centralissimo(nx.Graph(dense_adj))

    for epoch in parameter_config['epochs']:
        gamma = np.random.beta(1, 1.005 - parameter_config['basef'] ** epoch)
        alpha = beta = (1 - gamma) / 2
        optimizer.zero_grad()
        out = model(g_data.x, g_data.edge_index)
        prob = F.softmax(out).cpu().numpy()
        # prob是样本 * 类别
        # 这里的entropy函数计算的是一列的信息熵，所以我们要转置一下，让一列成类，计算一个样本的信息熵
        entropy = scipy.stats.entropy(prob.T)
        kmeans = KMeans(n_clusters=g_data.NCL, random_state=0).fit(prob)
        ed_score = euclidean_distances(prob, kmeans.cluster_centers_)
        density = np.min(ed_score, axis=1)
        # entropy和density的norm: 计算样本中的百分位数（因为只需要比较样本之间的分数即可）
        norm_entropy = [perc_for_entropy(entropy, i) for i in range(len(entropy))]
        norm_density = [perc_for_entropy(density, i) for i in range(len(density))]
        finalweight = alpha * norm_entropy + beta * norm_density + gamma * norm_centrality

        # 把train, val和test数据排除
        finalweight[g_data.train_mask + g_data.val_mask + g_data.test_mask] = -100
        select = np.argmax(finalweight)
        g_data.train_mask.append(select)



        loss = criterion(out[g_data.train_mask], g_data.y[g_data.train_mask])
        loss.backward()
        optimizer.step()

        print("Epoch {:}, training loss is {:.3f}, validation loss is {:.3f}, ")


adata = get_anndata()


