import torch
import torch.nn.functional as F
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

        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        attn = attn.softmax()  # (sparse) [N, N, nh]
        # attn = attn.to_dense().view(N, -1)        
        # attn = F.softmax(attn, dim=1)
        # # print(type(attn))
        # attn = dglsp.spmatrix(attn.nonzero().t(), attn[attn.nonzero(as_tuple=True)].reshape(1, -1))
        # print(attn.shape)
        # print(attn)
        # exit()
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]

        # attn = (q.transpose(0, 2).transpose(1, 2) @ k.transpose(0, 2)) * A.to_dense().to(device) # [nh, N, N]
        # attn = attn.softmax(dim=0)  # [nh, N, N]        
        # out = attn @ v.transpose(0, 2).transpose(1, 2) # [nh, N, dh]
        
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
        # h2 = h
        # h = self.FFN1(h)
        # h = F.relu(h)
        # h = F.dropout(h, self.dropout, training=self.training)
        # h = self.FFN2(h)

        # if self.residual:      
        #     h = h2 + h

        # h = self.batchnorm2(h)

        return h


class GTModel(nn.Module):
    def __init__(
            self,
            args,
            in_dim,            
            class_num,          
            pos_enc                      
    ):
        super().__init__()        
        self.n_classes = class_num
        self.in_dim = in_dim
        self.out_dim = args.out_dim
        self.residual = args.residual
        self.add_pos_enc = args.add_pos_enc
        self.hidden_dim = args.hidden_dim
        self.pos_enc_dim = args.pos_enc_dim
        self.num_heads = args.n_heads
        self.dropout_rate = args.dropout_rate
        self.num_layers = args.n_layers        
        self.pos_enc = pos_enc
        # self.task = args.task
        self.state_embeddings = None        
        self.use_auxilary = args.use_auxilary
        self.is_auxilary = args.is_auxilary
        self.h_embedding = nn.Linear(self.in_dim, self.hidden_dim)                        
        self.pos_linear = nn.Linear(self.pos_enc_dim, self.hidden_dim)
        self.layers = nn.ModuleList(
            [GTLayer(self.hidden_dim, self.hidden_dim, self.num_heads, self.dropout_rate, residual=self.residual) for _ in range(self.num_layers - 1)]            
        )        
        
        self.layers.append(
            GTLayer(self.hidden_dim, self.out_dim, self.num_heads, self.dropout_rate, residual=self.residual)
        )
        # self.pooler = dglnn.SumPooling()                
        self.predictor = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim // 2),
            nn.ReLU(),
            # nn.Linear(hidden_size // 2, hidden_size // 4),
            # nn.ReLU(),
            nn.Linear(self.out_dim // 2, self.n_classes),
        )
    
    def get_embeddings(self, g_data, args):
        self.eval()                
        # 这里edge_index 变成 torch.tensor([[srt...], [dst...]])                  
        print("type model adj shape is {:}".format(g_data.adjacency_matrix().to_dense().shape))      
        
        edge_index = torch.stack(g_data.edges())
        indices = edge_index.to(device)        
        features = g_data.ndata['x'].to(device)    
        if args.add_pos_enc: 
            pos_enc = g_data.ndata['PE'].to(device)
        N = features.shape[0] # N * feature_dim
        A = dglsp.spmatrix(indices, shape=(N, N))        
        # A = g.edges()
        h = self.h_embedding(features)            
        if args.add_pos_enc:
            h = h + self.pos_linear(pos_enc)
        
        for layer in self.layers:
            h = layer(A, h)
        return h
    
    def set_state_embeddings(self, embeddings):
        self.state_embeddings = embeddings.detach()     

    def forward(self, edge_index, features):
        indices = edge_index.to(device)
        N = features.shape[0] # N * feature_dim
        A = dglsp.spmatrix(indices, shape=(N, N))        
        # A = g.edges()
        h = self.h_embedding(features)            
        if self.add_pos_enc:
            h = h + self.pos_linear(self.pos_enc)
        
        for layer in self.layers:
            h = layer(A, h)
        
        if not self.is_auxilary and self.use_auxilary:
            h = h + self.state_embeddings
            h = self.predictor(h)
        else:
            h = self.predictor(h)
        # print(h.shape)
        # h = self.pooler(g, h)
        # exit()
        return h
