import pandas as pd
from scipy.sparse import load_npz
import numpy as np
def get_edge_idx(df):
    # 转置DataFrame，将行变为列          
    row = map(int, df.iloc[0, :].tolist())
    col = map(int, df.iloc[1, :].tolist())
    graph = set(tuple(zip(row, col)))    
    return graph
                  
    
old_graph = load_npz('old_graph.npz')
new_graph = load_npz('new_graph.npz') # 经过了normalization


print("new_graph的L1: ", new_graph.sum())
print("old_graph的L1：",old_graph.sum())

old_graph[old_graph > 0] = 1
new_graph[new_graph > 0] = 1

print("new_graph的L1: ", new_graph.sum())
print("old_graph的L1：",old_graph.sum())

graph = new_graph - old_graph
graph = graph.toarray()
# 统计-1的数量，删除的边
delete_num = np.count_nonzero(graph < 0)
new_num = np.count_nonzero(graph > 0)
print("新增边数{:}".format(new_num))
print("删除的边数{:}".format(delete_num))

