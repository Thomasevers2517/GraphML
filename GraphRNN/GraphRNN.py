import torch
import torch.sparse as sparse
from Neighbor_Agregation import Neighbor_Aggregation
        
class Graph_RNN(torch.nn.Module):
    def __init__(self, n_nodes, n_features, h_size, f_out_size, edge_weights=None):
        """ Initialize the Graph RNN
        Args:
            n_nodes (int): number of nodes in the graph
            n_features (int): number of features in the input tensor
            h_size (int): size of the hidden state
            f_size (int): size of the vector returned by the neighbor aggregation function
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_time_steps, n_edges, 3), per edge (node1 , node2, edge_feat)
            """
        super(Graph_RNN, self).__init__()
        assert f_out_size == h_size # For now, we assume that the output size of the RNN is the same as the hidden state size
        
        self.edge_weights = edge_weights
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.h_size = h_size
        self.f_out_size = f_out_size
        
        self.A = torch.nn.parameter.Parameter(torch.randn(h_size, h_size)) 
        self.B = torch.nn.parameter.Parameter(torch.randn(h_size, n_features))
        self.C = torch.nn.parameter.Parameter(torch.randn(h_size, f_out_size))
        
        self.D = torch.nn.parameter.Parameter(torch.randn(n_features, h_size))
        self.E = torch.nn.parameter.Parameter(torch.randn(h_size, h_size))
        

        self.AG = Neighbor_Aggregation(n_nodes, h_size, f_out_size, edge_weights= None)
        
        self.H  = None
        
        self.node_idx = None
     
    
    
    def forward(self, x_in, edge_weights=None, pred_hor = 1):
        if self.node_idx is None:
            self.node_idx = torch.zeros(self.n_nodes)
            self.node_idx = edge_weights[0, :, 0].unique() 
            
        self.H  = None

        x_out = torch.zeros((x_in.shape[0], pred_hor, x_in.shape[2], x_in.shape[3]))
        
        for i in range(x_in.shape[1]):
            if i< x_in.shape[1]:
                x_out[:,i,:,:] = self.forward_step(x_in[:, i, :], edge_weights=edge_weights[:,i,:,:])
            else:
                x_out[:,i,:,:] = self.forward_step(x_out[:, i-1, :], edge_weights=edge_weights[:,i,:,:])
        return x_out
        
    
    def forward_step(self, x_in, edge_weights=None):
        
        if edge_weights is None:
            if self.edge_weights is None:
                raise ValueError("Edge weights not provided. Provide edge weights to the forward pass or during initialization.")
            edge_weights = self.edge_weights
            
        if self.H is None:
            self.H = torch.zeros((x_in.shape[0], self.n_nodes, self.h_size, self.h_size))
       
        self.neigh_ag = self.AG(self.H, edge_weights=edge_weights, node_idx=self.node_idx)
        
        #stack A's and such!!
        self.H = torch.tanh(torch.matmul(self.A, self.H) + torch.matmul(self.B, x_in) + torch.matmul(self.C, self.neigh_ag) + self.D)    
        
        x_out = torch.matmul(self.E, self.H)
        
        return x_out