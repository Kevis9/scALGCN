import pandas as pd


def get_tuple(df):
    # 转置DataFrame，将行变为列
    transposed_df = df.transpose()
    
    graph = set()
    # 输出每一列作为一个元组
    for column in transposed_df.values:
        graph.add(tuple(map(int, column)))
    
    return graph
                  
    
    
old_graph = get_tuple(pd.read_csv('old_graph.csv'))
new_graph = get_tuple(pd.read_csv('new_graph.csv'))

# 新增的边
new_edge = new_graph - old_graph
print("新增边数{:}".format(len(new_edge)))
print(new_edge)

# 删除的边数
del_edge = old_graph - new_graph
print("删除的边数{:}".format(len(del_edge)))
print(del_edge)

