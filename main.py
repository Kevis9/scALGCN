import torch
import os
from utils import read_data_with_mm, capsule_pd_data_to_anndata, read_data_with_csv, combine_inter_intra_graph
import anndata as ad
import pandas as pd
# Data preparation

data_config = {
    'ref_data_path': 'ref_data',
    'query_data_path': 'query_data',
    'ref_label_path': 'ref_label',
    'query_label_path': 'query_label',
    'anndata_path': 'anndata.h5ad',
    'inter_graph_path': '',
    'intra_graph_path': '',
    'mode': 'csv'
}

parameter_config = {
    'gcn_hidden_units': 256,
    'epochs': 200,
}

root_data_path = 'experiment/Baron_Xin'
for key in data_config:
    if "path" in key:
        data_config[key] = os.path.join(root_data_path, data_config[key])


def get_anndata():
    if data_config['mode'] == 'csv':
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
        adata.write(os.path.join(root_data_path, 'data.h5ad'), compression="gzip")
        return adata
    elif data_config['mode'] == 'ann':
        '''ann data已存在'''
        adata = ad.read_hdf(data_config['anndata_path'])
        return adata


def train(model):
    model.train()
    criterion =
    for epoch in parameter_config['epochs']:
        model.zero_grad()




adata = get_anndata()


