# 对Graph transformer的适配
import dgl
import scipy.stats
import torch
import os
import torch_geometric.data
from utils import capsule_pd_data_to_anndata, combine_inter_intra_graph, \
    centralissimo, perc_for_entropy, perc_for_density, setup_seed, random_stratify_sample, \
    random_stratify_sample_with_train_idx, load_data

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
from model import GTModel, GraphTransformerModel


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

# For active learning and training
parameter_config = {
    'epochs': 200,
    'gt_lr': 1e-3,
    # For active learning
    'basef': 0.8,
    'k_select': 1,
    'NL': 100,  # 有标签节点选取的阈值，这里的初始值不重要，最后 = NC * 20, 按照论文里面的设置
    'wd': 5e-4,  # weight decay
    'initial_class_train_num': 10,
    'epoch_print_flag': True,
    'final_class_num': 30
}

# For GT hyper-parameters
net_params = {
    'in_dim': 0, # not sure now
    'hidden_dim': 128,
    'out_dim': 128,
    'n_classes': 0, # not sure now
    'n_heads': 1,    
    'dropout': 0.2,
    'n_layers': 3,
    'layer_norm': False,
    'batch_norm': True,    
    'residual': True,
    'add_pos_enc':True,
    'device': device,
    'lap_pos_enc': True,
    'wl_pos_enc': False,
    'pos_enc_dim': 8,
}


def train(model, g_data, data_info, is_active_learning):
    '''
        model: Graph transformer
        g_data: DGLGraph
        is_active_learning: for active learning
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

        g_data = g_data.to(device)
        node_x = g_data.ndata['x'].to(device)
        lap_pe = g_data.ndata['PE'].to(device)

        out = model(g_data, node_x, lap_pe)

        # out = model(g_data, node_x, h_lap_pos_enc=lap_pe)

        loss = criterion(out[data_info['train_idx']], g_data.ndata['y_predict'][data_info['train_idx']])
        loss.backward()
        optimizer.step()

        prob = F.softmax(out.detach(), dim=1).cpu().numpy()

        # prob是样本 * 类别
        # 这里的entropy函数计算的是一列的信息熵，所以我们要转置一下，让一列成类，计算一个样本的信息熵
        if is_active_learning and len(data_info['train_idx']) < parameter_config['NL']:
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

        test_pred = torch.argmax(out[data_info['test_idx']], dim=1).cpu().numpy()
        test_true = g_data.ndata['y_true'][data_info['test_idx']].cpu().numpy()
        test_acc = accuracy_score(test_pred, test_true)

        train_pred = torch.argmax(out[data_info['train_idx']], dim=1).cpu().numpy()
        train_true = g_data.ndata['y_true'][data_info['train_idx']].cpu().numpy()
        train_acc = accuracy_score(train_pred, train_true)

        if (parameter_config['epoch_print_flag']):
            print("Epoch {:}: traing loss: {:.3f}, train_acc: {:.3f}, val_loss {:.3f}, val_acc {:.3f}, test_acc {:.3f}".format(epoch, loss, train_acc, val_loss,
                                                                                           val_acc, test_acc))


    return model


def test(model, g_data, data_info):
    # model.eval()
    g_data.to(device)
    out = model(g_data.to(device), g_data.ndata["x"].to(device), g_data.ndata["PE"].to(device))
    test_pred = torch.argmax(out[data_info['test_idx']], dim=1).cpu().numpy()
    test_true = g_data.ndata['y_true'][data_info['test_idx']].cpu().numpy()
    test_acc = accuracy_score(test_true, test_pred)
    print("test acc {:.3f}".format(test_acc))
    return test_acc


projects = [
    'seq_well_10x_v3'
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
    
    # load data
    g_data, adata, data_info = load_data(data_config=data_config, parameter_config=parameter_config)
    
    # 设置好NL的值
    parameter_config['NL'] = data_info['NCL'] * parameter_config['final_class_num']
    g_data.ndata['PE'] = dgl.laplacian_pe(g_data, k=net_params['pos_enc_dim'], padding=True)

    # ours    
    model = GTModel(input_dim=g_data.ndata['x'].shape[1], 
                    n_class=data_info['NCL'], 
                    hidden_size=net_params['hidden_dim'],
                    out_dim=net_params['out_dim'], 
                    pos_enc_size=net_params['pos_enc_dim'],
                    num_layers=net_params['n_layers'],
                    drop_out=net_params['dropout'],
                    num_heads=net_params['n_heads'],
                    residual=net_params['residual'],
                    add_pos_enc=net_params['add_pos_enc']).to(device)
    
    train(model, g_data, data_info, is_active_learning=False)    
    test_acc = test(model, g_data, data_info)
    AL_acc.append(test_acc)
    AL_ref_num.append(len(data_info['train_idx']))

    # Graph Transformer
    data_info['train_idx'] = adata.uns['train_idx_for_no_al']
    # model = GTModel(input_dim=g_data.ndata['x'].shape[1],
    #                 out_size=data_info['NCL'],
    #                 hidden_size=parameter_config['hidden_size'],
    #                 num_heads=parameter_config['num_heads'],
    #                 num_layers=parameter_config['num_layers'],
    #                 pos_enc_size=parameter_config['pos_enc_size']).to(device)
    #
    # net_params['in_dim'] = g_data.ndata['x'].shape[1]
    # net_params['n_classes'] = data_info['NCL']
    # model = GraphTransformerModel(net_params)

    # train(model, g_data, data_info, is_active_learning=False)
    # test_acc = test(model, g_data, data_info)
    # gt_acc.append(test_acc )
    # gt_ref_num.append(len(data_info['train_idx']))


    query_num_arr.append(len(data_info['test_idx']))
    data_config = data_config_cp
    
    print(projects)
    print(AL_acc)
    print(AL_ref_num)
    print(gt_acc)
    print(gt_ref_num)
    print(query_num_arr)
