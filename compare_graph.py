import pandas as pd


def get_edge_idx(df):
    # 转置DataFrame，将行变为列          
    row = map(int, df.iloc[0, :].tolist())
    col = map(int, df.iloc[1, :].tolist())
    graph = set(tuple(zip(row, col)))    
    return graph
                  
    
    
old_graph = get_edge_idx(pd.read_csv('old_graph.csv', header=None))
new_graph = get_edge_idx(pd.read_csv('new_graph.csv', header=None))

# 新增的边
new_edge = new_graph - old_graph
print("新增边数{:}".format(len(new_edge)))
# print(new_edge)

# 删除的边数
del_edge = old_graph - new_graph
print("删除的边数{:}".format(len(del_edge)))
# print(del_edge)

