import pandas as pd
from scipy.sparse import load_npz
import numpy as np
import argparse
import os
def get_edge_idx(df):
    # 转置DataFrame，将行变为列          
    row = map(int, df.iloc[0, :].tolist())
    col = map(int, df.iloc[1, :].tolist())
    graph = set(tuple(zip(row, col)))    
    return graph
                  

parser = argparse.ArgumentParser()
# data config
parser.add_argument('--res_path', type=str, 
                             default='/home/hwl/scALGCN/result/bcp1_6000-bcp2_6000-exp0013_GT + GL', 
                             help='data directory')
args = parser.parse_args()

res_path = args.res_path

old_graph = load_npz(os.path.join(res_path, 'old_graph.npz'))
new_graph = load_npz(os.path.join(res_path, 'new_graph.npz')) # 经过了normalization


print("new_graph的L1: ", new_graph.sum())
print("old_graph的L1：",old_graph.sum())

old_graph[old_graph > 0] = 1
new_graph[new_graph > 0] = 1

print("after > 0 set to 1")
print("new_graph的L1: ", new_graph.sum())
print("old_graph的L1：",old_graph.sum())

graph = new_graph - old_graph
graph = graph.toarray()
# 统计-1的数量，删除的边
delete_num = np.count_nonzero(graph < 0)
new_num = np.count_nonzero(graph > 0)
print("新增边数{:}".format(new_num))
print("删除的边数{:}".format(delete_num))

