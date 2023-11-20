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
from graphtransformer.layers.graph_transformer_layer import GraphTransformerLayer
from graphtransformer.layers.mlp_readout_layer import MLPReadout

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scaling = self.out_dim ** -0.5
        

        self.q_proj = nn.Linear(in_dim, out_dim * self.num_heads)
        self.k_proj = nn.Linear(in_dim, out_dim * self.num_heads)
        self.v_proj = nn.Linear(in_dim, out_dim * self.num_heads)
        

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.out_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.out_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.out_dim, self.num_heads)

        # >>>>> Using sparse matrix to compute
        # >>>>> There is a problem in dglsp.bsdmm: the program will crash without any prompts
        # attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # >>>>> Instead, we will use normal dense computation
        attn = (q.transpose(0, 2).transpose(1, 2) @ k.transpose(0, 2)) * A.to_dense().to(device) # [nh, N, N]
        attn = attn.softmax(dim=0)  # [nh, N, N]
        # out = dglsp.bspmm(attn, v)  # [N, dh, nh]
        out = attn @ v.transpose(0, 2).transpose(1, 2) # [nh, N, dh]
        
        return out.reshape(N, -1)


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, in_dim, out_dim, num_heads, dropout, residual):
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.MHA = SparseMHA(in_dim=in_dim, out_dim=out_dim, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(out_dim)
        self.batchnorm2 = nn.BatchNorm1d(out_dim)
        
        self.O = nn.Linear(out_dim, out_dim)
        self.FFN1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN2 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, A, h):
        # residual
        h1 = h
        h = self.MHA(A, h)

        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        if self.residual:
            h = h + h1
        
        h = self.batchnorm1(h)
        
        # residual 
        h2 = h
        h = self.FFN1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN2(h)

        if self.residual:      
            h = h2 + h

        h = self.batchnorm2(h)

        return h


class GTModel(nn.Module):
    def __init__(
            self,
            input_dim,
            out_dim,
            n_class,
            hidden_size,           
            pos_enc_size,
            num_layers,
            drop_out,
            residual,
            num_heads
    ):
        super().__init__()        
        self.h_embedding = nn.Linear(input_dim, hidden_size)
        self.hiddens_size = hidden_size        
        self.n_class = n_class
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.residual = residual
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.layers = nn.ModuleList(
            [GTLayer(hidden_size, hidden_size, num_heads, self.drop_out, residual=residual) for _ in range(num_layers - 1)]
        )        
        self.layers.append(
            GTLayer(hidden_size, out_dim, num_heads, self.drop_out, residual=residual)
        )
        # self.pooler = dglnn.SumPooling()                
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            # nn.Linear(hidden_size // 2, hidden_size // 4),
            # nn.ReLU(),
            nn.Linear(out_dim // 2, n_class),
        )
                

    def forward(self, g, X, pos_enc):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))        
        # A = g.edges()
        h = self.h_embedding(X) + self.pos_linear(pos_enc)
        for layer in self.layers:
            h = layer(A, h)

        # print(h.shape)
        # h = self.pooler(g, h)
        # exit()
        return self.predictor(h)


class GraphTransformerModel(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['n_layers']

        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 100

        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                                           dropout, self.layer_norm, self.batch_norm, self.residual) for
                                     _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e=None, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)

        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)

        # output
        h_out = self.MLP_layer(h)

        return h_out

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss