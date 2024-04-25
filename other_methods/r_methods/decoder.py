import pandas as pd
import anndata as ad
import argparse
import os
from scipy.sparse import csr_matrix
from scipy.io import mmread
from sklearn.metrics import accuracy_score, f1_score
import re

methods = ['seurat', 'scmap', 'chetah', 'singler']

parser = argparse.ArgumentParser()
parser.add_argument('--proj', type=str, default='')

args = parser.parse_args()
proj = args.proj

acc_data = pd.read_csv('result/acc.csv', index_col=0)
f1_data = pd.read_csv('result/f1_macro.csv', index_col=0)

for i, method in enumerate(methods):
    query_pred = pd.read_csv(os.path.join('result', proj, method, 'query_pred.csv')).to_numpy()
    query_true = pd.read_csv(os.path.join('result', proj, method, 'query_true.csv')).to_numpy()
    acc = accuracy_score(query_pred, query_true)
    f1_macro = f1_score(query_true, query_pred, average='macro')    
    if proj not in acc_data.index.tolist():
        new_row = {col: '' for col in acc_data.columns}
        acc_data.loc[proj] = new_row
    
    if proj not in f1_data.index.tolist():
        new_row = {col: '' for col in f1_data.columns}
        f1_data.loc[proj] = new_row
    
    acc_data.loc[proj][method]=acc
    f1_data.loc[proj][method]=f1_macro

acc_data.to_csv('result/acc.csv')
f1_data.to_csv('result/f1_macro.csv')