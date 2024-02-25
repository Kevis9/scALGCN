import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pgd import PGD, prox_operators
import warnings
import networkx as nx
from utils import centralissimo, accuracy, active_learning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score

class ProGNN:
    """ ProGNN (Properties Graph Neural Network). See more details in Graph Structure Learning for Robust Graph Neural Networks, KDD 2020, https://arxiv.org/abs/2005.10203.

    Parameters
    ----------
    model:
        model: The backbone GNN model in ProGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
    See details in https://github.com/ChandlerBang/Pro-GNN.

    """

    def __init__(self, model, args, data_info, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None        
        self.weights = None
        self.estimator = None
        self.model = model.to(device)
        self.data_info = data_info       
         

    def fit(self, g_data):
        """Train Pro-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
                
        args = self.args    
        self.model_optimizer = optim.Adam(self.model.parameters(),
                               lr=args.gt_lr, weight_decay=args.wd)
        
        # 不需要转为numpy        
        save_eidx = torch.stack(g_data.edges()).cpu().numpy()
        np.savetxt('old_graph.csv', save_eidx, delimiter=',')        
        adj = g_data.adjacency_matrix().to_dense().to(self.device)
        estimator = EstimateAdj(adj, symmetric=args.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.model_optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=args.adj_lr)

        self.model_optimizer_l1 = PGD(estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=args.adj_lr, alphas=[args.alpha])

        warnings.warn("If you find the nuclear proximal operator runs too slow, you can modify line 77 to use prox_operators.prox_nuclear_cuda instead of prox_operators.prox_nuclear to perform the proximal on GPU. See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        self.model_optimizer_nuclear = PGD(estimator.parameters(),
                  proxs=[prox_operators.prox_nuclear_cuda],
                  lr=args.adj_lr, alphas=[args.beta])


        node_x = g_data.ndata['x'].to(self.device)
        labels = g_data.ndata['y_true'].to(self.device)

        if args.is_auxilary:
            train_idx = self.data_info['auxilary_train_idx']
            val_idx = self.data_info['auxilary_val_idx']            
        else:
            train_idx = self.data_info['train_idx']
            val_idx = self.data_info['val_idx']
        
        
        if args.is_auxilary:
            criterion = torch.nn.MSELoss()
        else:            
            criterion = torch.nn.CrossEntropyLoss()
            
        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            
            if args.adj_training:
                # Update S
                for i in range(int(args.outer_steps)):
                    self.train_adj(epoch, node_x, adj, labels,
                            train_idx, val_idx)
                         
            updated_adj = self.estimator.get_estimated_adj()            
            # after updating S, need to calculate the norm centrailty again for selecting new nodes
            # ======= graph active learning ======                    
            if args.active_learning:
                graph = nx.Graph(updated_adj.detach().cpu().numpy())
                norm_centrality = centralissimo(graph) 
            
            for i in range(int(args.inner_steps)):
                prob = self.train_gnn(adj=updated_adj, 
                               features=node_x,                               
                               labels=labels,
                               epoch=epoch,
                               criterion=criterion,
                               edge_index=torch.stack(g_data.edges()).to(self.device))
                                
                
                if args.active_learning:
                    # will change outer data_info (the parameter is reference)
                    active_learning(g_data=g_data,
                                    epoch=epoch,
                                    out_prob=prob,
                                    norm_centrality=norm_centrality,
                                    args=self.args,
                                    data_info=self.data_info)
                    
                                        

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))                

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

    def train_gnn(self, adj, features, labels, epoch, criterion, edge_index=None):
        args = self.args
        labels = labels.to(self.device)
        if args.debug:
            print("\n=== train_gnn ===")                
        
        # adj = adj.detach().clone()
        # # 尝试调整adj的阈值
        # adj[adj < self.args.adj_thresh] = 0              
        # edge_index = adj.nonzero().T
        
                
        t = time.time()
        self.model.train()
        self.model_optimizer.zero_grad()
        # GTModel        
        output = self.model(edge_index, features)
        
        if args.is_auxilary:
            train_idx = self.data_info['auxilary_train_idx']
            val_idx = self.data_info['auxilary_val_idx']
        else:
            train_idx = self.data_info['train_idx']
            val_idx = self.data_info['val_idx']
                
        loss_train = criterion(output[train_idx], labels[train_idx])
        if not args.is_auxilary:
            # main model
            acc_train = accuracy(output[train_idx], labels[train_idx])
        
        loss_train.backward()
        self.model_optimizer.step()
        
        prob = F.softmax(output.detach(), dim=1).cpu().numpy()                        
                
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(edge_index, features)

        loss_val = criterion(output[val_idx], labels[val_idx])
        if args.is_auxilary:
            if loss_val < self.best_val_loss:
                self.best_val_loss = loss_val
                self.best_graph = adj.detach()
                self.weights = deepcopy(self.model.state_dict())
                if self.args.debug:
                    print(f'saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())
        else:            
            acc_val = accuracy(output[val_idx], labels[val_idx])
            if acc_val > self.best_val_acc:
                self.best_val_acc = acc_val
                self.best_graph = adj.detach()
                self.weights = deepcopy(self.model.state_dict())
                if self.args.debug:
                    print(f'saving current model and graph, best_val_acc: %s' % self.best_val_acc.item())
        if self.args.debug:  
            if args.is_auxilary:          
                print('Epoch: {:04d}'.format(epoch+1),
                        'loss_train: {:.4f}'.format(loss_train.item()),
                        'loss_val: {:.4f}'.format(loss_val.item()),
                        'time: {:.4f}s'.format(time.time() - t))
            else:
                print('Epoch: {:04d}'.format(epoch+1),
                        'loss_train: {:.4f}'.format(loss_train.item()),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'loss_val: {:.4f}'.format(loss_val.item()),
                        'acc_val: {:.4f}'.format(acc_val.item()),
                        'time: {:.4f}s'.format(time.time() - t))
                

        return prob

    def train_adj(self, epoch, features, original_adj, labels, idx_train, idx_val):        
        estimator = self.estimator
        args = self.args
        # adj = estimator.get_estimated_adj()
        
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.model_optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - original_adj, p='fro')        
        estimated_adj = estimator.get_estimated_adj().clone().detach()
        
        if args.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        
        estimated_adj[estimated_adj < self.args.adj_thresh] = 0
        edge_index = estimated_adj.nonzero().T        
        
        if args.is_auxilary:
            criterion = torch.nn.MSELoss()
            
        else:
            criterion = torch.nn.CrossEntropyLoss()    
            
        output = self.model(edge_index, features)
        
        loss_gcn = criterion(output[idx_train], labels[idx_train])
        
        if not args.is_auxilary:
            # if is main model
            acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_symmetric = torch.norm(estimator.estimated_adj \
                        - estimator.estimated_adj.t(), p="fro")

        loss_diffiential =  loss_fro + args.gamma * loss_gcn + args.lambda_ * loss_smooth_feat + args.phi * loss_symmetric

        loss_diffiential.backward()

        self.model_optimizer_adj.step()
        loss_nuclear =  0 * loss_fro
        if args.beta != 0:
            self.model_optimizer_nuclear.zero_grad()
            self.model_optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.model_optimizer_l1.zero_grad()
        self.model_optimizer_l1.step()

        total_loss = loss_fro \
                    + args.gamma * loss_gcn \
                    + args.alpha * loss_l1 \
                    + args.beta * loss_nuclear \
                    + args.phi * loss_symmetric
        
        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))
        
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        estimated_adj = estimator.get_estimated_adj().clone().detach()
        estimated_adj[estimated_adj < self.args.adj_thresh] = 0
        edge_index = estimated_adj.nonzero().T           
        output = self.model(edge_index, features)

        loss_val = criterion(output[idx_val], labels[idx_val])
        if args.is_auxilary:
            if loss_val < self.best_val_loss:
                self.best_val_loss = loss_val
                self.best_graph = estimated_adj.detach()
                self.weights = deepcopy(self.model.state_dict())
                if args.debug:
                    print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())
        else:
            acc_val = accuracy(output[idx_val], labels[idx_val])                
            print('Epoch: {:04d}'.format(epoch+1),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))
            
            if acc_val > self.best_val_acc:
                self.best_val_acc = acc_val
                self.best_graph = estimated_adj.detach()
                self.weights = deepcopy(self.model.state_dict())
                if args.debug:
                    print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())            

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj-original_adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))


    def test(self, features, idx_test, labels, edge_index=None):
        """
            Evaluate the performance of ProGNN on test set
        """
        print("\t=== testing ===")
        
        if self.args.is_auxilary:
            criterion = torch.nn.MSELoss()        
        else:
            criterion = torch.nn.CrossEntropyLoss()            
            
        self.model.eval()
        adj = self.best_graph
        if self.best_graph is None:
            adj = self.estimator.get_estimated_adj()
        
        adj[adj < self.args.adj_thresh] = 0
        # edge_index = adj.nonzero().T                 
        
        output = self.model(edge_index, features)                
        save_eidx = edge_index.detach().cpu().numpy()
        np.savetxt('new_graph.csv', save_eidx, delimiter=',')

        loss_test = criterion(output[idx_test], labels[idx_test])
        if self.args.is_auxilary:
            print("\tTest set results:",
                "loss= {:.4f}".format(loss_test.item())
                )
            y_pred = output[idx_test].detach().cpu().numpy()
            y_true = labels[idx_test].detach().cpu().numpy()
            mse = mean_squared_error(y_pred, y_true)
            mae = mean_absolute_error(y_pred, y_true)
            r2 = r2_score(y_pred, y_true)
            print("scALGT regression mse: {:.3f}, mae: {:.3f}, r2: {:.3f}".format(mse, mae, r2))            
            return loss_test.item()
        else:            
            acc_test = accuracy(output[idx_test], labels[idx_test])
            macrof1_test = f1_score(output[idx_test].detach().cpu().numpy(), labels[idx_test].detach().cpu().numpy(), average='macro')
            
            print("\tTest set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()))
            return acc_test.item(), macrof1_test, output[idx_test].detach().cpu().numpy() 
        

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())/2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

    def get_estimated_adj(self):
        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())/2
        else:
            adj = self.estimated_adj
        
        return adj
