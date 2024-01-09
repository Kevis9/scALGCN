from utils import load_data
import argparse
from regression_methods import Regressor

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, 
                             default='D:/YuAnHuang/kevislin/scALGCN/experiments/EXP0001/data', 
                             help='data directory')

parser.add_argument('--task', type=str,
                            default='cell state',
                            help='"cell type" or "cell state"')

parser.add_argument('--init_train_num', type=int, 
                             default=10, 
                             help='for active learning, we will pick some initial nodes for training')

args = parser.parse_args() 

_, adata, data_info = load_data(args)

model = Regressor(adata.X, data_info['train_idx'], data_info['test_idx'], adata.uns['cell_type'])

model.linear_regression()
model.lasso()
model.random_forest()
model.svr()
