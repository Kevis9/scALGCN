from scipy.io import mmread
import pandas as pd

def read_mm_data(path):
    '''
    Matrix Market Format gene expression data
    :param path: data path
    :return: A Dense Numpy matrix with cell * gene
    '''
    data = mmread(path).to_dense()
    return data
def read_data_with_csv(data_path, label_path):
    '''
    :param data_path:
    :param label_path:
    :return: Tuple of gene expression matrix(Numpy), label (Numpy)
    '''
    data = pd.read_csv(data_path, index_col=0)
    label = pd.read_csv(label_path)
    data = data.to_numpy()
    label = label.to_numpy()
    return data, label

def read_data_with_mm(data_path, label_path):
    data = mmread(data_path).to_dense()
    label = pd.read_csv(label_path).to_numpy()
    return data, label

def read_adjacent_matrix(data_path):
    '''
    :param data_path: Adjacent matrix in the format of matrix market
    :return: dense matrix of numpy for a graph (adjacent matrix)
    '''
    A = mmread(data_path).to_dense()
    return A

