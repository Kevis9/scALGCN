import dgl
import os
from utils import load_data
import numpy as np
import torch
from model import GTModel
import json
import argparse
from prognn import ProGNN
import pandas as pd
import copy
from utils import setup_seed

setup_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


parser.add_argument('--debug', action='store_true', 
                             default=True, 
                             help='debug mode')

parser.add_argument('--hidden_dim', type=int,
                             default=256, 
                             help='hidden dim for graph transformer')
parser.add_argument('--n_heads', type=int,
                             default=2, 
                             help='num of heads for GTModel')
parser.add_argument('--n_layers', type=int, 
                             default=3, 
                             help='num of layers for GTModel')
parser.add_argument('--pos_enc_dim', type=int,
                             default=8, 
                             help='positional encoding dim')

####### Graph Learning #######
parser.add_argument('--GL_epochs', type=int, 
                             default=5 , #一般设置为5或者10 <epochs最好
                             help='epochs for GL')

parser.add_argument('--alpha', type=float, 
                    default=2, 
                    help='weight of l1 norm')
parser.add_argument('--beta', type=float, 
                    default=1.5, 
                    help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, 
                    default=1, 
                    help='weight of GNN loss')
parser.add_argument('--lambda_', type=float, 
                    default=5e-4, 
                    help='weight of feature smoothing')



####### Switchs #######
parser.add_argument('--layer_norm', action='store_true',
                             default=False, 
                             help='layer norm for GTModel')
parser.add_argument('--batch_norm', action='store_true',
                             default=True, 
                             help='Batch norm for GTModel')
parser.add_argument('--residual', action='store_true',
                             default=True, 
                             help='residual for GTModel')
parser.add_argument('--symmetric', action='store_true', 
                            default=True,
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
                    default=False,
                    help='for GTModel, whether use auxilary model')


args = parser.parse_args()

proj = args.data_dir.split('/')[-1]
args.data_dir = os.path.join(args.data_dir, 'data')

# load data
g_data, auxilary_g_data, adata, data_info = load_data(args=args, use_auxilary=args.use_auxilary)
max_nodes_num = data_info['class_num'] * args.max_per_class
data_info['max_nodes_num'] = max_nodes_num

# For debug information
print("data path is {:}, \n ref_data num: {:}, \nquery_data num :{:}, \n auxilary data num:{:}".format(args.data_dir, adata.uns['n_ref'], adata.uns['n_query'], auxilary_g_data.num_nodes() if args.use_auxilary else 0))



if args.use_auxilary:        
    # auxilary model no need: AL and GL
    auxilary_args = copy.copy(args)
    auxilary_args.active_learning = False
    auxilary_args.updated_adj = False
    auxilary_model = GTModel(args=auxilary_args,                    
                    class_num=data_info['auxilary_class_num'],
                    in_dim=auxilary_g_data.ndata['x'].shape[1],
                    pos_enc=auxilary_g_data.ndata['PE'].to(device) if args.add_pos_enc else None).to(device)

    # use Pro-GNN to train the GT
    auxilary_model_prognn = ProGNN(auxilary_model, data_info=data_info, args=auxilary_args, device=device)
    auxilary_model_prognn.fit(g_data=auxilary_g_data)



'''
 ========= For cell type prediction ========= 
'''
args.is_auxilary = False
type_model = GTModel(args=args,                
                class_num=data_info['class_num'],
                in_dim=auxilary_g_data.ndata['x'].shape[1],
                pos_enc=g_data.ndata['PE'].to(device) if args.add_pos_enc else None).to(device)

if args.use_auxilary:
    auxilary_embeddings = auxilary_model.get_embeddings(g_data=g_data, args=args)
    type_model.set_state_embeddings(auxilary_embeddings)

prognn = ProGNN(type_model, data_info=data_info, args=args, device=device)
prognn.fit(g_data=g_data)


test_res = prognn.test(features=g_data.ndata['x'].to(device), 
                       idx_test=data_info['test_idx'], 
                       labels=g_data.ndata['y_true'].to(device))


print("acc is {:.3f}".format(test_res[0]))


# save config
ref_proj = proj.split('-')[0]
query_proj = proj.split('-')[1]
auxilary_proj = proj.split('-')[2]

if not args.use_auxilary:
    auxilary_proj = ''

with open('config/{:}-{:}-{:}_acc_{:.3f}.json'.format(ref_proj, query_proj, auxilary_proj, test_res[0]), 'w') as f:
    json.dump(vars(args), f)
    
second_key = 'GT'
if args.add_pos_enc:
    second_key += ' + pos'

if args.active_learning:
    second_key += ' + AL'

if args.adj_training:
    second_key += ' + GL'
    
first_key = ref_proj + '-' + query_proj
if args.use_auxilary:
    first_key += ('-' + auxilary_proj)

print("experimens {:}_{:} finished".format(first_key, second_key))
acc_data = pd.read_csv('result/acc.csv', index_col=0)
acc_data.loc[first_key][second_key] = test_res[0]
acc_data.to_csv('result/acc.csv')

f1_data = pd.read_csv('result/macro-f1.csv', index_col=0)
f1_data.loc[first_key][second_key] = test_res[1]
f1_data.to_csv('result/macro-f1.csv')

# save query_true.csv, query_predict.csv
ref_true = data_info['label_encoder'].inverse_transform(g_data.ndata['y_true'].numpy()[:adata.uns['n_ref']])
query_true = data_info['label_encoder'].inverse_transform(g_data.ndata['y_true'].numpy()[adata.uns['n_ref']:])
query_predict = data_info['label_encoder'].inverse_transform(test_res[2])

ref_true_df = pd.DataFrame(ref_true, columns=['cell_type'])
query_true_df = pd.DataFrame(query_true, columns=['cell_type'])
query_predict_df = pd.DataFrame(query_predict, columns=['cell_type'])

exp_save_path = os.path.join('result', first_key + '_' + second_key)
if not os.path.exists(exp_save_path):
    os.makedirs(exp_save_path)

ref_true_df.to_csv(os.path.join(exp_save_path, 'ref_true.csv'), index=False)
query_true_df.to_csv(os.path.join(exp_save_path, 'query_true.csv'), index=False)
query_predict_df.to_csv(os.path.join(exp_save_path, 'query_pred.csv'), index=False)

# save ref embeedings and query embeedings and auxilary embeddings
if args.use_auxilary:
    np.save(os.path.join(exp_save_path, 'auxilary_embeddings.npy'), auxilary_embeddings.detach().cpu().numpy())

ref_query_embeddings = type_model.get_embeddings(g_data=g_data, args=args).detach().cpu().numpy()
ref_embeddings = ref_query_embeddings[:adata.uns['n_ref']]
query_embeddings = ref_query_embeddings[adata.uns['n_ref']:]

np.save(os.path.join(exp_save_path, 'ref_embeddings.npy'), ref_embeddings)
np.save(os.path.join(exp_save_path, 'query_embeddings.npy'), query_embeddings)

