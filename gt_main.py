import dgl
import os
from utils import setup_seed, load_data, train, test
import numpy as np
import torch
from model import GTModel
import json
import argparse
from prognn import ProGNN
import pandas as pd
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# data config
parser.add_argument('--data_dir', type=str, 
                             default='/home/Users/kevislin/scALGCN/experiments/bcp1_6000-bcp2_6000-exp0013', 
                             help='data directory')
####### Active learning #######
parser.add_argument('--basef', type=float, 
                             default=0.8, 
                             help='base factor for active learning')
parser.add_argument('--k_select', type=int, 
                             default=1, 
                             help='num of nodes to select for every iteration')
parser.add_argument('--init_train_num', type=int, 
                             default=100, 
                             help='for active learning, we will pick some initial nodes for training')
parser.add_argument('--max_per_class', type=int, 
                             default=30, 
                             help='max number of nodes for each class')

####### GT Model #######
parser.add_argument('--epochs', type=int, 
                             default=30, 
                             help='epochs for training')

parser.add_argument('--wd', type=float, 
                             default=0.0005, 
                             help='weight decay')
parser.add_argument('--debug', action='store_true', 
                             default=True, 
                             help='debug mode')

parser.add_argument('--gt_lr', type=float,
                             default=1e-3, 
                             help='learning rate for graph transformer')
parser.add_argument('--hidden_dim', type=int,
                             default=256, 
                             help='hidden dim for graph transformer')
parser.add_argument('--out_dim', type=int,
                             default=128, 
                             help='output dim of GTModel, input dim for classifier')
parser.add_argument('--n_heads', type=int,
                             default=1, 
                             help='num of heads for GTModel')
parser.add_argument('--dropout_rate', type=float,
                             default=0.2, 
                             help='dropout rate for GTModel')
parser.add_argument('--n_layers', type=int, 
                             default=2, 
                             help='num of layers for GTModel')
parser.add_argument('--pos_enc_dim', type=int,
                             default=8, 
                             help='positional encoding dim')

####### Graph Learning #######
parser.add_argument('--adj_lr', type=float,
                             default=1e-3, 
                             help='learning rate for training adj')
parser.add_argument('--alpha', type=float, 
                    default=1e-4, 
                    help='weight of l1 norm')
parser.add_argument('--beta', type=float, 
                    default=1e-3, 
                    help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, 
                    default=1, 
                    help='weight of GNN loss')
parser.add_argument('--lambda_', type=float, 
                    default=1e-5, 
                    help='weight of feature smoothing')

parser.add_argument('--phi', type=float, 
                    default=0,
                    help='weight of symmetric loss')

parser.add_argument('--inner_steps', type=int, 
                    default=2, 
                    help='steps for inner optimization')

parser.add_argument('--outer_steps', type=int, 
                    default=1, 
                    help='steps for outer optimization')
parser.add_argument('--adj_thresh', type=float,
                            default=1e-3,
                            help='threshold for adj to turn to 0')

####### Switch #######

parser.add_argument('--layer_norm', action='store_true',
                             default=False, 
                             help='layer norm for GTModel')
parser.add_argument('--batch_norm', action='store_true',
                             default=True, 
                             help='Batch norm for GTModel')
parser.add_argument('--residual', action='store_true',
                             default=False, 
                             help='residual for GTModel')

parser.add_argument('--symmetric', action='store_true', 
                            default=False,
                            help='whether use symmetric matrix')


parser.add_argument('--adj_training', action='store_true',
                    default=False,
                    help='whether update the adj')

parser.add_argument('--add_pos_enc', action='store_true',
                             default=False, 
                             help='whether adding postional encoding to node feature')

parser.add_argument('--active_learning', action='store_true', 
                             default=False, 
                             help='active learning mode')

parser.add_argument('--is_auxilary', action='store_true',
                    default=True,
                    help='is auxilari model?')

parser.add_argument('--use_auxilary', action='store_true',
                    default=True,
                    help='for GTModel, whether use auxilary model')

parser.add_argument('--exp_reverse', action='store_true',
                    default=False,
                    help='reverse ref_data and query_data')


args = parser.parse_args()

proj = args.data_dir.split('/')[-1]
args.data_dir = os.path.join(args.data_dir, 'data')
seed = 32
setup_seed(seed)

    
# load data
g_data, auxilary_g_data, adata, data_info = load_data(args=args, use_auxilary=args.use_auxilary)
max_nodes_num = data_info['class_num'] * args.max_per_class
data_info['max_nodes_num'] = max_nodes_num

# For debug information
print("data path is {:}, \n ref_data num: {:}, \nquery_data num :{:}, \n auxilary data num:{:}".format(args.data_dir, adata.uns['n_ref'], adata.uns['n_query'], auxilary_g_data.num_nodes()))

if args.add_pos_enc:
    g_data.ndata['PE'] = dgl.lap_pe(g_data, k=args.pos_enc_dim, padding=True)
    auxilary_g_data.ndata['PE'] = dgl.lap_pe(auxilary_g_data, k=args.pos_enc_dim, padding=True)

if args.use_auxilary:
    auxilary_model = GTModel(args=args,
                    in_dim=auxilary_g_data.ndata['x'].shape[1],
                    class_num=data_info['auxilary_class_num'],
                    pos_enc=auxilary_g_data.ndata['PE'].to(device) if args.add_pos_enc else None).to(device)

    # use Pro-GNN to train the GT
    auxilary_model_prognn = ProGNN(auxilary_model, data_info=data_info, args=args, device=device)
    auxilary_model_prognn.fit(g_data=auxilary_g_data)



'''
 ========= For cell type prediction ========= 
'''
args.is_auxilary = False
type_model = GTModel(args=args,
                in_dim=g_data.ndata['x'].shape[1],
                class_num=data_info['class_num'],
                pos_enc=g_data.ndata['PE'].to(device) if args.add_pos_enc else None).to(device)

if args.use_auxilary:
    auxilary_embeddings = auxilary_model.get_embeddings(g_data=g_data, args=args)
    type_model.set_state_embeddings(auxilary_embeddings)

prognn = ProGNN(type_model, data_info=data_info, args=args, device=device)
prognn.fit(g_data=g_data)

test_res = prognn.test(features=g_data.ndata['x'].to(device), 
                       idx_test=data_info['test_idx'], 
                       labels=g_data.ndata['y_true'].to(device), 
                       edge_index=torch.stack(g_data.edges()).to(device))


print("acc is {:.3f}".format(test_res))


# save config
ref_proj = proj.split('-')[0]
query_proj = proj.split('-')[1]
auxilary_proj = proj.split('-')[2]

if args.exp_reverse:
    tmp = ref_proj
    ref_proj = query_proj
    query_proj = tmp

if not args.use_auxilary:
    auxilary_proj = ''

with open('config/{:}-{:}-{:}_acc_{:.3f}.json'.format(ref_proj, query_proj, auxilary_proj, test_res), 'w') as f:
    json.dump(vars(args), f)
    
acc_data = pd.read_csv('result/acc.csv', index_col=0)
second_key = 'GT'
if args.add_pos_enc:
    second_key += ' + pos'

if args.active_learning:
    second_key += ' + AL'

if args.adj_training:
    second_key += ' + GL'
    
last_proj = ref_proj + '-' + query_proj
if args.use_auxilary:
    last_proj += ('-' + auxilary_proj)

acc_data.loc[last_proj][second_key] = test_res
acc_data.to_csv('result/acc.csv')
    
    

