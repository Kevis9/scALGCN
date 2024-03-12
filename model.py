import torch
import torch.nn.functional as F
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaling = hidden_dim ** -0.5
        

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)        
        # 这里把A先当做dense matrix来看待
        # bmm: (b, n, m) * (b, m, q) = (m, n, q)
        # q.transpose(0, 2).transpose(1, 2) = (nh, N, dh)
        # k.transpose(0, 2) = (nh, dh, N)
        # attn = (nh, N, N)
        attn = torch.mul(A, torch.bmm(q.transpose(0, 2).transpose(1, 2), k.transpose(0, 2)))        
        # attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        # attn = attn.softmax()  # (sparse) [N, N, nh]
        attn = F.softmax(attn, dim=2)
        # v.transpose(0, 2).transpose(1, 2) = [nh,N,dh]
        # attn = [nh, N, N]
        # out = [nh, N, dh]
        out = torch.bmm(attn, v.transpose(0, 2).transpose(1, 2))        
        # out = [N, dh, nh]
        out = out.transpose(0, 1).transpose(1, 2)
        # out = dglsp.bspmm(attn, v)  # [N, dh, nh]
        out = self.out_proj(out.reshape(N, -1))
        return out


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_dim, num_heads, residual):
        super().__init__()        
        self.residual = residual
        self.MHA = SparseMHA(hidden_dim=hidden_dim, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
                
        self.FFN1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.FFN2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, A, h):
        # attention-layer
        h1 = h
        h = self.MHA(A, h)
        
        if self.residual:
            h = h + h1  

        h = self.batchnorm1(h)

        h2 = h
        # two-layer FNN
        h = self.FFN2(F.relu(self.FFN1(h)))        

        if self.residual:      
            h = h2 + h
        h = self.batchnorm2(h)
        return h


class GTModel(nn.Module):
    def __init__(
            self,
            args,                                                       
            class_num,          
            in_dim,
            pos_enc                      
    ):
        super().__init__()        
        self.n_classes = class_num
        self.residual = args.residual
        self.add_pos_enc = args.add_pos_enc
        self.hidden_dim = args.hidden_dim
        self.pos_enc_dim = args.pos_enc_dim
        self.num_heads = args.n_heads        
        self.num_layers = args.n_layers        
        self.pos_enc = pos_enc
        self.in_dim = in_dim
        # self.task = args.task
        self.state_embeddings = None        
        self.use_auxilary = args.use_auxilary
        self.is_auxilary = args.is_auxilary
        # 论文中使用的nn.Embedding， 原因是输入的数据是词嵌入（整数类型），而我们这里只需要直接映射就好
        self.h_embedding = nn.Linear(self.in_dim, self.hidden_dim)                        
        self.pos_linear = nn.Linear(self.pos_enc_dim, self.hidden_dim)
        self.layers = nn.ModuleList(
            [GTLayer(self.hidden_dim, self.num_heads, residual=self.residual) for _ in range(self.num_layers)]            
        )                                  
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.n_classes),
        )
    
    def get_embeddings(self, g_data, args):
        self.eval()                
        # 这里edge_index 变成 torch.tensor([[srt...], [dst...]])                                  
        # edge_index = torch.stack(g_data.edges())
        # indices = edge_index.to(device)        
        A = g_data.adj.todense().to(device)
        features = g_data.ndata['x'].to(device)    
        if args.add_pos_enc: 
            pos_enc = g_data.ndata['PE'].to(device)
        N = features.shape[0] # N * feature_dim        
        # A = dglsp.spmatrix(indices, shape=(N, N))                        
        h = self.h_embedding(features)            
        if args.add_pos_enc:
            h = h + self.pos_linear(pos_enc)
        
        for layer in self.layers:
            h = layer(A, h)

        if not args.is_auxilary and args.use_auxilary and self.state_embeddings is not None:
            h = h + self.state_embeddings
        
        return h
    
    def set_state_embeddings(self, embeddings):
        self.state_embeddings = embeddings.detach()     

    def forward(self, A, features):
        # A是一个dense的邻接矩阵
        # indices = edge_index.to(device)
        # N = features.shape[0] # N * feature_dim
        # A = dglsp.spmatrix(indices, shape=(N, N))        
        # A = g.edges()
        h = self.h_embedding(features)            
        if self.add_pos_enc:
            h = h + self.pos_linear(self.pos_enc)
        
        for layer in self.layers:
            h = layer(A, h)        
        
        if not self.is_auxilary and self.use_auxilary:
            h = h + self.state_embeddings
                                                                
        h = self.predictor(h)        
        return h
