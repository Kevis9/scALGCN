from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm

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
        adj = SparseTensor(row=edge_idx[0], col=edge_idx[1], sparse_sizes=(x.shape[0], x.shape[0]))
        x = self.conv1(x, adj.t())
        x = x.relu()
        # x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, adj.t())
        return x


class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        attn = attn.softmax()  # (sparse) [N, N, nh]
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]

        return self.out_proj(out.reshape(N, -1))


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, A, h):
        h1 = h
        h = self.MHA(A, h)
        h = self.batchnorm1(h + h1)

        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        h = h2 + h

        return self.batchnorm2(h)

class GTModel(nn.Module):
    def __init__(
        self,
        input_dim,
        out_size,
        hidden_size=80,
        pos_enc_size=2,
        num_layers=8,
        num_heads=8,
    ):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_size)
        self.h_embedding = nn.Linear(input_dim, hidden_size)
        self.hiddens_size = hidden_size
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.layers = nn.ModuleList(
            [GTLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.pooler = dglnn.SumPooling()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, out_size),
        )

    def forward(self, g, X, pos_enc):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))
        h = self.h_embedding(X) + self.pos_linear(pos_enc)
        for layer in self.layers:
            h = layer(A, h)
        h = self.pooler(g, h)

        return self.predictor(h)

