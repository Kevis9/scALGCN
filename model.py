from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

class GCN(torch.nn.Module):
    '''
        Two layer GCN network
    '''
    def __init__(self, input_dim, hidden_units, output_dim):
        super(GCN, self).__init__()
        torch.manual_seed(32)
        self.conv1 = GCNConv(input_dim, hidden_units)
        self.conv2 = GCNConv(hidden_units, output_dim)

    def forward(self, x, edge_idx):
        x = self.conv1(x, edge_idx)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_idx)
        return x


