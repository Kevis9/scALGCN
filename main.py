import scipy.stats
import torch
import os

import torch_geometric.data
from utils import capsule_pd_data_to_anndata, combine_inter_intra_graph, \
    centralissimo, perc_for_entropy, perc_for_density

from model import GCN
import anndata as ad
import pandas as pd
import torch.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
from torch_geometric.utils import to_dense_adj
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# device = 'gpu' if torch.device
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
    'label_rate': 0.2,
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

        # 随机分层采样
        adata.uns['train_idx'], adata.uns['val_idx'] = random_stratify_sample(ref_label.numpy())

        # test_idx即query data的idx
        adata.uns['test_idx'] = [len(ref_label) + i for i in range(len(query_label))]
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
        model.train()

        gamma = np.random.beta(1, 1.005 - parameter_config['basef'] ** epoch)
        alpha = beta = (1 - gamma) / 2
        optimizer.zero_grad()
        out = model(g_data.x, g_data.edge_index)
        prob = F.softmax(out).cpu().numpy()

        loss = criterion(out[g_data.train_idx], g_data.y_predict[g_data.train_idx])
        loss.backward()
        optimizer.step()

        # prob是样本 * 类别
        # 这里的entropy函数计算的是一列的信息熵，所以我们要转置一下，让一列成类，计算一个样本的信息熵
        entropy = scipy.stats.entropy(prob.T)
        kmeans = KMeans(n_clusters=g_data.NCL, random_state=0).fit(prob)
        ed_score = euclidean_distances(prob, kmeans.cluster_centers_)
        density = np.min(ed_score, axis=1)
        # entropy和density的norm: 计算样本中的百分位数（因为只需要比较样本之间的分数即可）
        norm_entropy = [perc_for_entropy(entropy, i) for i in range(len(entropy))]
        norm_density = [perc_for_density(density, i) for i in range(len(density))]
        finalweight = alpha * norm_entropy + beta * norm_density + gamma * norm_centrality

        # 把train, val的数据排除
        finalweight[g_data.train_idx + g_data.val_idx] = -100
        select = np.argmax(finalweight)

        # 每一个epoch就增加一个节点，前提是预测的准确率大于设置的置信度
        # 不考虑原论文的budget，感觉没用
        if prob[select].max() >= 0.6 and select not in g_data.train_idx:
            g_data.train_idx.append(select)
            # 注意y_predict是tensor
            g_data.y_predict[select] = prob[select].argmax()
            print("Epoch {:}: pick up one node to the training set!".format(epoch))

            # validation
        model.eval()
        val_pred = out[g_data.val_idx].argmax()
        val_acc = accuracy_score(val_pred.cpu().numpy(), g_data.y_true[g_data.val_idx].cpu().numpy())
        val_loss = criterion(out[g_data.val_idx], g_data.y_true[g_data.val_idx])

        print("Epoch {:}: traing loss: {:.3f}, val_loss {:.3f}, val_acc {:.3f}".format(epoch, loss, val_loss, val_acc))
        # todo: 加入早停法

        return model


def test(model, g_data):
    model.eval()
    out = model(g_data.x, g_data.edge_index)
    test_pred = out[g_data.test_idx].argmax()
    test_acc = accuracy_score(test_pred.cpu().numpy(), g_data.y_true[g_data.test_idx].cpu().numpy())
    print("test acc {:.3f}".format(test_acc))


def random_stratify_sample(ref_labels):
    # 对每个类都进行随机采样，分成train, val
    # 这地方要保证train的数据是从0开始计数的,
    label_set = set(list(ref_labels))
    train_idx = []
    val_idx = []
    for c in label_set:
        idx = np.where(ref_labels == c)[0]
        np.random.shuffle(idx)
        train_num = int(0.8 * len(ref_labels))
        train_idx += list(idx[:train_num])
        val_idx += list(idx[train_num:])
    return train_idx, val_idx


# 数据准备
adata = get_anndata()

# g_data准备
# 将所有的adata的数据转为tensor
g_data = torch_geometric.data.Data(x=torch.tensor(adata.x, dtype=torch.float),
                                   edge_index=torch.tensor(adata.uns['edge_index'], dtype=torch.long))


y_true = adata.obs['cell_type']
label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(y_true)
g_data.y_true = torch.tensor(y_true, dtype=torch.long)
g_data.y_predict = torch.tensor(y_true, dtype=torch.long)
g_data.val_idx = adata.uns['val_idx']
g_data.test_idx = adata.uns['test_idx']
g_data.NCL = len(set(adata.obs['cell_type'][adata.uns['train_idx']]))

label_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
for rate in label_rate:
    #todo: 对于随机性实验，为了严谨考虑，要多做几组
    g_data.train_idx = adata.uns['train_idx'][:int(rate*len(adata.uns['train_idx']))]
    model = GCN(input_dim=g_data.x.shape[1], hidden_units=parameter_config['gcn_hidden_units'], output_dim=g_data.NCL)
    train(model, g_data)
