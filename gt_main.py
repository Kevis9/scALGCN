import dgl
import os
from utils import setup_seed, load_data, train, test
import numpy as np
import torch
from model import GTModel
import json
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = {
    'data_config' :{
        'root': 'experiment/baron_xin/data',
        'ref_data_path': 'ref_data.csv',
        'query_data_path': 'query_data.csv',
        'ref_label_path': 'ref_label.csv',
        'query_label_path': 'query_label.csv',
        'anndata_path': 'data.h5ad',
        'inter_graph_path': 'inter_graph.csv',
        'intra_graph_path': 'intra_graph.csv',
        'data_mode': 'ann'
    },
    'para_config' :{
        'epochs': 200,        
        # For active learning
        'basef': 0.8,
        'k_select': 1,
        'NL': 100,  # 有标签节点选取的阈值，这里的初始值不重要，最后 = NC * 20, 按照论文里面的设置
        'wd': 5e-4,  # weight decay
        'initial_class_train_num': 10,
        'epoch_print_flag': True,
        'multi_class_num': 30, # NL = NC * multi_class_num
        'is_active_learning': True,
        'early_stop': True,
        'tolerance_epoch': 30, 

        # GT hyper-parameters
        'gt_lr': 1e-3,
        'in_dim': 0, # not sure now
        'hidden_dim': 256,
        'out_dim': 128,
        'n_classes': 0, # not sure now
        'n_heads': 1,    
        'dropout_rate': 0.2,
        'n_layers': 2,
        'pos_enc_dim': 8,
        'layer_norm': False,
        'batch_norm': True,    
        'residual': False,
        'add_pos_enc': True,                     
    }
}

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
    config['data_config']['root'] = 'experiment/' + proj + '/data'
    root_data_path = config['data_config']['root']
    data_config_cp = config['data_config'].copy()
    for key in config['data_config']:
        if "path" in key:
            config['data_config'][key] = os.path.join(root_data_path, config['data_config'][key])
    
    # load data
    g_data, adata, data_info = load_data(data_config=config['data_config'], parameter_config=config['para_config'])
    
    
    # 设置好NL的值
    config['para_config']['NL'] = data_info['NCL'] * config['para_config']['multi_class_num']        
    g_data.ndata['PE'] = dgl.laplacian_pe(g_data, k=config['para_config']['pos_enc_dim'], padding=True)

    # set right parameters
    config['para_config']['n_classes'] = data_info['NCL']
    config['para_config']['in_dim'] = g_data.ndata['x'].shape[1]
    
    
    # ours    
    model = GTModel(config['para_config']).to(device)
    
    model = train(model, g_data, data_info, config)    
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
    config['data_config'] = data_config_cp

    # save config
    with open('{:}_acc_{:.3f}.json'.format(proj, test_acc), 'w') as f:
        json.dump(config, f)

results = dict(zip(projects, AL_acc))
print(results)
    
    