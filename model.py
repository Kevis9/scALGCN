from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
class GCN(torch.nn.Module):
    '''
        Two layer GCN network
    '''
    def __init__(self, input_dim, hidden_units, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_units)
        self.conv2 = GCNConv(hidden_units, output_dim)

    def forward(self, x, edge_idx):
        adj = SparseTensor(row=edge_idx[0], col=edge_idx[1], sparse_sizes=(x.shape[0], x.shape[0]))
        x = self.conv1(x, adj.t())
        x = x.relu()
        # x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, adj.t())
        return x


