import os
import random
import numpy.random
import scipy.stats
import torch
import dgl
import scipy.sparse as sp
import pandas as pd
import numpy as np
import anndata as ad
import networkx as nx
from scipy.io import mmread
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    # random.seed(seed)
    torch.cuda.manual_seed_all(seed) #所有GPU
    torch.cuda.manual_seed(seed)     # 当前GPU
    # CUDA有些算法是non deterministic, 需要限制    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
    torch.use_deterministic_algorithms(False)
    

def capsule_pd_data_to_anndata(data, label, edge_index):
    '''
    :param data: pandas frame, including cell names, gene names
    :param label: cell labels (pandas)
    :param edge_index: COO format [[row...], [col...]]
    :return:
    '''
    adata = ad.AnnData(data.to_numpy(), dtype=float)
    adata.obs_names = data.index.tolist()
    adata.var_names = data.columns.tolist()
    adata.obs['cell_type'] = label.iloc[:, 0].to_numpy()
    adata.uns['edge_index'] = edge_index
    return adata


def median_normalization(data):
    '''
    :param data: matrix of (cells * genes)
    :return: median normalized data
    '''
    row_sum = np.sum(data, axis=1)
    mean_transcript = np.mean(row_sum)
    print("细胞平均表达量是 {:.3f}".format(mean_transcript))
    row_sum[np.where(row_sum == 0)] = 1
    # scale_factor = 1e4
    # data_norm = np.log1p((data / row_sum.reshape(-1 ,1))*scale_factor)
    data_norm = (data / row_sum.reshape(-1, 1)) * mean_transcript
    return data_norm

def combine_inter_intra_graph(inter_graph_path, intra_graph_path, n_nodes_ref, n_nodes_query):
    '''
    combine inter-graph and intra-graph to get hybrid graoh and convert it to coo format(edge_index)
    :param inter_graph_path:
    :param intra_graph_path:
    :return:
    '''
    inter_graph = pd.read_csv(inter_graph_path, index_col=0)
    intra_graph = pd.read_csv(intra_graph_path, index_col=0)

    # 先对cell2的数目做一个映射
    inter_graph['V2'] += n_nodes_ref
    intra_graph['V1'] += n_nodes_ref
    intra_graph['V2'] += n_nodes_ref

    # 获取row和col
    row = inter_graph['V1'].tolist() + intra_graph['V1'].tolist()
    col = inter_graph['V2'].tolist() + intra_graph['V2'].tolist()

    # 构建一个adj矩阵（保证是对称矩阵，是无向图)
    adj = np.identity(n_nodes_ref+n_nodes_query)
    adj[row, col] = 1
    adj[col, row] = 1

    # 再转成COO format
    row, col = adj.nonzero()
    row = list(row)
    col = list(col)

    return np.array([row, col])


#graph central
def centralissimo(G):
    centralities = []
    centralities.append(nx.pagerank(G))
    L = len(centralities[0])
    Nc = len(centralities)
    cenarray = np.zeros((Nc,L))
    for i in range(Nc):
        cenarray[i][list(centralities[i].keys())]=list(centralities[i].values())
    normcen = (cenarray.astype(float)-np.min(cenarray,axis=1)[:,None])/(np.max(cenarray,axis=1)-np.min(cenarray,axis=1))[:,None]
    return normcen


# calculate the percentage of elements smaller than the k-th element
def perc_for_entropy(input, k):
    return sum([1 if i else 0 for i in input < input[k]]) / float(len(input))


# calculate the percentage of elements larger than the k-th element
def perc_for_density(input, k): return sum([1 if i else 0 for i in input > input[k]]) / float(len(input))


def random_stratify_sample(ref_labels, train_size):
    # 对每个类都进行随机采样，分成train, val
    # 这地方要保证train的数据是从0开始计数的,
    # print(ref_labels.squeeze())
    label_set = set(list(ref_labels.squeeze()))
    train_idx = []
    val_idx = []
    for c in label_set:
        idx = np.where(ref_labels == c)[0]
        np.random.seed(20)
        np.random.shuffle(idx)
        train_num = int(train_size * len(idx))
        train_idx += list(idx[:train_num])
        val_idx += list(idx[train_num:])

    return train_idx, val_idx


def random_stratify_sample_with_train_idx(ref_labels, train_idx, train_class_num):
    '''
    paramters:
        train_idx: 训练集下标
        train_class_num: 训练数据集中类别的数目
    '''
    label_set = list(set(list(ref_labels.squeeze())))
    label_set.sort()
    new_train_idx = []
    for c in label_set:
        idx = np.array(train_idx)[np.where(ref_labels[train_idx] == c)[0]]
        np.random.seed(20)
        if len(idx) < train_class_num:
            random_nodes = list(np.random.choice(idx, len(idx), replace=False))
        else:
            random_nodes = list(np.random.choice(idx, train_class_num, replace=False))
        new_train_idx += random_nodes
    return new_train_idx


def get_anndata(data_config, parameter_config):
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

        adata.write(os.path.join(data_config['root'], 'data.h5ad'), compression="gzip")
        return adata
    elif data_config['data_mode'] == 'ann':
        '''ann data已存在'''
        adata = ad.read(data_config['anndata_path'])
        return adata



def load_data(data_config, parameter_config):
    # 数据准备
    adata = get_anndata(data_config=data_config, parameter_config=parameter_config)
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
    data_info['label_encoder'] = label_encoder
    return g_data, adata, data_info


def train(model, g_data, data_info, config):
    '''
        model: Graph transformer
        g_data: DGLGraph        
    '''
    model.to(device)
    g_data = g_data.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['para_config']['gt_lr'],
                                 weight_decay=config['para_config']['wd'])

    dense_adj = g_data.adjacency_matrix().to_dense().cpu().numpy()
    graph = nx.Graph(dense_adj)
    norm_centrality = centralissimo(graph)
    # # 已选取的节点数目

    max_val_acc = 0
    tolerance_epoch = 0
    model_cp = model.copy()
    
    for epoch in range(config['para_config']['epochs']):
        model.train()

        gamma = np.random.beta(1, 1.005 - config['para_config']['basef'] ** epoch)
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
        if config['para_config']['is_active_learning'] and len(data_info['train_idx']) < config['para_config']['NL']:
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
            select_arr = np.argpartition(finalweight, -config['para_config']['k_select'])[-config['para_config']['k_select']:]
            for node_idx in select_arr:
                data_info['train_idx'].append(node_idx)
                # 注意y_predict是tensor
                g_data.ndata['y_predict'][node_idx] = g_data.ndata['y_true'][node_idx]
                if (config['para_config']['epoch_print_flag']):
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

        if (config['para_config']['epoch_print_flag']):
            print("Epoch {:}: traing loss: {:.3f}, train_acc: {:.3f}, val_loss {:.3f}, val_acc {:.3f}, test_acc {:.3f}".format(epoch, loss, train_acc, val_loss,
                                                                                           val_acc, test_acc))
        if (config['para_config']['early_stop']):
            if max_val_acc < val_acc:
                tolerance_epoch = 0
                max_val_acc = val_acc
                model_cp = model.copy()
            else:                
                tolerance_epoch += 1
                if tolerance_epoch > config['para_config']['tolerance_epoch']:
                    print("Early stop at epoch {:}, return the max_val_acc model.".format(epoch))
                break
            

    return model_cp


def test(model, g_data, data_info):
    # model.eval()
    g_data.to(device)
    out = model(g_data.to(device), g_data.ndata["x"].to(device), g_data.ndata["PE"].to(device))
    test_pred = torch.argmax(out[data_info['test_idx']], dim=1).cpu().numpy()
    test_true = g_data.ndata['y_true'][data_info['test_idx']].cpu().numpy()
    test_acc = accuracy_score(test_true, test_pred)
    print("test acc {:.3f}".format(test_acc))
    return test_acc
