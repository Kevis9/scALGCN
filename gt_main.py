# 对Graph transformer的适配
import dgl
import scipy.stats
import torch
import os
import torch_geometric.data
from utils import capsule_pd_data_to_anndata, combine_inter_intra_graph, \
    centralissimo, perc_for_entropy, perc_for_density, setup_seed, random_stratify_sample, \
    random_stratify_sample_with_train_idx

from model import GCN
import anndata as ad
import pandas as pd
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
from torch_geometric.utils import to_dense_adj
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from model import GTModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_config = {
    'root': 'experiment/baron_xin/data',
    'ref_data_path': 'ref_data.csv',
    'query_data_path': 'query_data.csv',
    'ref_label_path': 'ref_label.csv',
    'query_label_path': 'query_label.csv',
    'anndata_path': 'anndata.h5ad',
    'inter_graph_path': 'inter_graph.csv',
    'intra_graph_path': 'intra_graph.csv',
    'data_mode': 'csv'
}

# For Graph Transformer parameters
parameter_config = {
    'gcn_hidden_units': 64,
    'epochs': 200,
    'gt_lr': 1e-2,
    'basef': 0.8,
    'k_select': 1,
    'NL': 100,  # 有标签节点选取的阈值，这里的初始值不重要，最后 = NC * 20, 按照论文里面的设置
    'wd': 5e-4,  # weight decay
    'initial_class_train_num': 10,
    'epoch_print_flag': False,
    'final_class_num': 30
}


def get_anndata():
    if data_config['data_mode'] == 'csv':
        ref_data = pd.read_csv(data_config['ref_data_path'], index_col=0)
        ref_label = pd.read_csv(data_config['ref_label_path'])
        query_data = pd.read_csv(data_config['query_data_path'], index_col=0)
        query_label = pd.read_csv(data_config['query_label_path'])

        data = pd.concat([ref_data, query_data], axis=0)
        label = pd.concat([ref_label, query_label], axis=0)
        edge_index = combine_inter_intra_graph(inter_graph_path=data_config['inter_graph_path'],
                                               intra_graph_path=data_config['intra_graph_path'],
                                               n_nodes_ref=len(ref_label),
                                               n_nodes_query=len(query_label))

        adata = capsule_pd_data_to_anndata(data, label, edge_index)

        # 随机分层采样
        adata.uns['train_idx'], adata.uns['val_idx'] = random_stratify_sample(ref_label.to_numpy(), train_size=0.8)
        adata.uns['train_idx_for_no_al'] = adata.uns['train_idx']
        # test_idx即query data的idx
        adata.uns['test_idx'] = [len(ref_label) + i for i in range(len(query_label))]

        # 按照论文，train label一开始每个类别设置成4个, 剩余的training node作为label budget里面的一部分
        adata.uns['train_idx'] = random_stratify_sample_with_train_idx(ref_label.to_numpy(),
                                                                       train_idx=adata.uns['train_idx'],
                                                                       train_class_num=parameter_config[
                                                                           'initial_class_train_num'])

        adata.write(os.path.join(root_data_path, 'data.h5ad'), compression="gzip")
        return adata
    elif data_config['data_mode'] == 'ann':
        '''ann data已存在'''
        adata = ad.read(data_config['anndata_path'])
        return adata


def train(model, g_data, data_info, select_mode):
    '''
        model: Graph transformer
        g_data: DGLGraph
        select_mode: for active learning
    '''
    model.to(device)
    g_data = g_data.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=parameter_config['gt_lr'],
                                 weight_decay=parameter_config['wd'])

    dense_adj = g_data.adjacency_matrix().to_dense().cpu().numpy()
    graph = nx.Graph(dense_adj)
    norm_centrality = centralissimo(graph)
    # # 已选取的节点数目

    for epoch in range(parameter_config['epochs']):
        model.train()

        gamma = np.random.beta(1, 1.005 - parameter_config['basef'] ** epoch)
        alpha = beta = (1 - gamma) / 2
        optimizer.zero_grad()
        out = model(g_data.to(device), g_data.ndata["x"].to(device), g_data.ndata["PE"].to(device))
       
        
        loss = criterion(out[data_info['train_idx']], g_data.ndata['y_predict'][data_info['train_idx']])
        loss.backward()
        optimizer.step()

        prob = F.softmax(out.detach(), dim=1).cpu().numpy()

        # prob是样本 * 类别
        # 这里的entropy函数计算的是一列的信息熵，所以我们要转置一下，让一列成类，计算一个样本的信息熵
        if select_mode and len(data_info['train_idx']) < parameter_config['NL']:
            entropy = scipy.stats.entropy(prob.T)
            kmeans = KMeans(n_clusters=data_info['NCL'], random_state=0).fit(prob)
            ed_score = euclidean_distances(prob, kmeans.cluster_centers_)
            density = np.min(ed_score, axis=1)
            # entropy和density的norm: 计算样本中的百分位数（因为只需要比较样本之间的分数即可）
            norm_entropy = np.array([perc_for_entropy(entropy, i) for i in range(len(entropy))])
            norm_density = np.array([perc_for_density(density, i) for i in range(len(density))])
            norm_centrality = norm_centrality.squeeze()
            finalweight = alpha * norm_entropy + beta * norm_density + gamma * norm_centrality

            # 把train, val, test的数据排除, 从剩余的label budget里面获取节点
            finalweight[data_info['train_idx'] + data_info['train_idx'] + data_info['val_idx']] = -100
            select_arr = np.argpartition(finalweight, -parameter_config['k_select'])[-parameter_config['k_select']:]
            for node_idx in select_arr:
                data_info['train_idx'].append(node_idx)
                # 注意y_predict是tensor
                g_data.ndata['y_predict'][node_idx] = g_data.ndata['y_true'][node_idx]
                if (parameter_config['epoch_print_flag']):
                    print("Epoch {:}: pick up one node to the training set!".format(epoch))

            # validation
        model.eval()
        # 做一个detach
        out = out.detach()
        val_pred = torch.argmax(out[data_info['val_idx']], dim=1).cpu().numpy()
        val_true = g_data.ndata['y_true'][data_info['val_idx']].cpu().numpy()
        val_acc = accuracy_score(val_true, val_pred)
        val_loss = criterion(out[data_info['val_idx']], g_data.ndata['y_true'][data_info['val_idx']])
        if (parameter_config['epoch_print_flag']):
            print("Epoch {:}: traing loss: {:.3f}, val_loss {:.3f}, val_acc {:.3f}".format(epoch, loss, val_loss,
                                                                                           val_acc))

    return model


def test(model, g_data, data_info):
    model.eval()
    g_data.to(device)
    out = model(g_data.to(device), g_data.ndata["x"].to(device), g_data.ndata["PE"].to(device))
    test_pred = torch.argmax(out[data_info['test_idx']], dim=1).cpu().numpy()
    test_true = g_data.ndata['y_true'][data_info['test_idx']].cpu().numpy()
    test_acc = accuracy_score(test_true, test_pred)
    print("test acc {:.3f}".format(test_acc))
    return test_acc


projects = [
    'cel_seq_10x_v3'
]

AL_acc = []
AL_ref_num = []
gt_acc = []
gt_ref_num = []
query_num_arr = []
scGCN_acc = []
scGCN_ref_num = []

seed = 32
setup_seed(seed)

for proj in projects:
    data_config['root'] = 'experiment/' + proj + '/data'
    root_data_path = data_config['root']
    data_config_cp = data_config.copy()
    for key in data_config:
        if "path" in key:
            data_config[key] = os.path.join(root_data_path, data_config[key])
    # 数据准备
    adata = get_anndata()
    # g_data准备
    # 将所有的adata的数据转为tensor
    # g_data = torch_geometric.data.Data(x=torch.tensor(adata.X, dtype=torch.float),
    #                                    edge_index=torch.tensor(adata.uns['edge_index'], dtype=torch.long))
    src, dst = adata.uns['edge_index'][0], adata.uns['edge_index'][1]
    g_data = dgl.graph((src, dst), num_nodes=len(adata.obs_names))

    y_true = adata.obs['cell_type']
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(y_true)
    g_data.ndata['x'] = torch.tensor(adata.X, dtype=torch.float)
    g_data.ndata['y_true'] = torch.tensor(y_true, dtype=torch.long)
    g_data.ndata['y_predict'] = torch.tensor(y_true, dtype=torch.long)

    data_info = {}
    data_info['val_idx'] = adata.uns['val_idx']
    data_info['test_idx'] = adata.uns['test_idx']
    data_info['train_idx'] = adata.uns['train_idx']
    data_info['NCL'] = len(set(adata.obs['cell_type'][adata.uns['train_idx']]))

    # 设置好NL的值
    parameter_config['NL'] = data_info['NCL'] * parameter_config['final_class_num']

    # ours
    # pos_enc_size = 8
    # g_data.ndata['PE'] = dgl.laplacian_pe(g_data, k=pos_enc_size, padding=True)
    # model = GTModel(input_dim=g_data.ndata['x'].shape[1], out_size=data_info['NCL'], pos_enc_size=pos_enc_size).to(device)
    # train(model, g_data, data_info, select_mode=True)
    #
    # test_acc = test(model, g_data, data_info)
    # AL_acc.append(test_acc)
    # AL_ref_num.append(len(data_info['train_idx']))

    # Graph Transformer
    data_info['train_idx'] = adata.uns['train_idx_for_no_al']
    model = GTModel(input_dim=g_data.ndata['x'].shape[1], out_size=data_info['NCL'], pos_enc_size=pos_enc_size).to(device)
    train(model, g_data, data_info, select_mode=False)
    test_acc = test(model, g_data, data_info)
    gt_acc.append(test_acc)
    gt_ref_num.append(len(data_info['train_idx']))


    query_num_arr.append(len(data_info['test_idx']))
    data_config = data_config_cp
    print(projects)
    print(AL_acc)
    print(AL_ref_num)
    print(gt_acc)
    print(gt_ref_num)
    print(query_num_arr)
