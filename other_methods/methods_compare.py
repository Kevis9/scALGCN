from utils import load_data
import argparse
from other_methods.regression_methods import Regressor
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                             default='D:/YuAnHuang/kevislin/scALGCN/experiments/EXP0047_EXP0071/data', 
                             help='data directory')

parser.add_argument('--task', type=str,
                            default='cell state',
                            help='"cell type" or "cell state"')

parser.add_argument('--init_train_num', type=int, 
                             default=100, # 完全不影响, 这里用的是全部的train样本, 没有val
                             help='for active learning, we will pick some initial nodes for training')

args = parser.parse_args() 

_, adata, data_info = load_data(args)

data_info['train_idx'] = adata.uns['original_train_idx']

model = Regressor(adata.X, data_info['train_idx'], data_info['test_idx'], adata.uns['cell_type'])

if not os.path.isfile('traditional_mse_res.csv'):
    mse_df = pd.DataFrame(columns=['linear', 'lasso', 'random_forest', 'svr'])
else:
    mse_df = pd.read_csv('traditional_mse_res.csv')
    
if not os.path.isfile('traditional_mae_res.csv'):
    mae_df = pd.DataFrame(columns=['linear', 'lasso', 'random_forest', 'svr'])
else:
    mae_df = pd.read_csv('traditional_mae_res.csv')

res1 = model.linear_regression()
res2 = model.lasso()
res3 = model.random_forest()
res4 = model.svr()

mse_series = pd.DataFrame({'linear':res1[0], 'lasso':res2[0], 'random_forest':res3[0], 'svr':res4[0]}, index=[args.data_dir.split('/')[-2]])

mae_series = pd.DataFrame({'linear':res1[1], 'lasso':res2[1], 'random_forest':res3[1], 'svr':res4[1]}, index=[args.data_dir.split('/')[-2]])


mse_df = pd.concat([mse_df, mse_series], ignore_index=False)
mae_df = pd.concat([mae_df, mae_series], ignore_index=False)

mse_df.to_csv("traditional_mse_res.csv")
mae_df.to_csv("traditional_mae_res.csv")

print(mse_df)
print(mae_df)