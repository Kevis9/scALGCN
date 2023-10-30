import random

import numpy.random
from scipy.io import mmread
import scipy.sparse as sp
import pandas as pd
import numpy as np
import anndata as ad
import networkx as nx
import torch
import os



def setup_seed(seed):
    torch.manual_seed(seed)
    # random.seed(seed)
    torch.cuda.manual_seed_all(seed) #所有GPU
    torch.cuda.manual_seed(seed)     # 当前GPU
    # CUDA有些算法是non deterministic, 需要限制    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
    torch.use_deterministic_algorithms(True)
    
def dense2sparse(data_path):
    '''
    transform dense matrix to sparse matrix of matrix market format
    :param data_path:
    :return:
    '''
    data = pd.read_csv(data_path, index_col=0)

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


def read_data_with_csv(data_path, label_path):
    '''
    read dense matrix with csv format
    :param data_path:
    :param label_path:
    :return: Tuple of gene expression matrix(Numpy), label (Numpy)
    '''
    data = pd.read_csv(data_path, index_col=0)
    label = pd.read_csv(label_path)
    data = data.to_numpy()
    label = label.to_numpy()
    return data, label


def read_data_with_mm(data_path, label_path):
    '''
    read compressed sparse matrix with matrix market format
    :param data_path:
    :param label_path:
    :return:
    '''
    data = mmread(data_path).to_dense()
    label = pd.read_csv(label_path).to_numpy()
    return data, label


def read_adjacent_matrix(data_path):
    '''
    :param data_path: Adjacent matrix in the format of matrix market
    :return: dense matrix of numpy for a graph (adjacent matrix)
    '''
    A = mmread(data_path).to_dense()
    return A


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

    # 构建一个adj矩阵（保证是对称矩阵）
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



