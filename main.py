import torch
import os
from utils import read_data_with_mm
import anndata as ad
# Data preparation

data_config = {
    'ref_data_path': 'ref_data',
    'query_data_path': 'query_data',
    'ref_label_path': 'ref_label',
    'query_label_path': 'query_label',
    'anndata_path': 'anndata.h5ad',
}

parameter_config = {
    'gcn_hidden_units': 256
}

root_data_path = 'experiment/Baron_Xin'
for key in data_config:
    if "path" in key:
        data_config[key] = os.path.join(root_data_path, data_config[key])


adata = ad.read_hdf(data_config['anndata_path'])
# ref_data, ref_label = read_data_with_mm(data_config['ref_data_path'], data_config['ref_label_path'])
# query_data, query_label = read_data_with_mm(data_config['query_data_path'], data_config['query_label_path'])

def train(model):
    model.train()
