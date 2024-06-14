import torch
import torch.sparse as sparse
from Neighbor_Agregation import Neighbor_Aggregation
from tqdm import tqdm
class Graph_RNN(torch.nn.Module):
    def __init__(self, n_nodes, n_features, h_size, f_out_size, edge_weights=None , device='cpu'):
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
        self.device = device
        
        self.edge_weights = edge_weights
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.h_size = h_size
        self.f_out_size = f_out_size
        
        self.init_H = torch.nn.parameter.Parameter(torch.randn(h_size)* 0.05, requires_grad=True)
        
        self.A = torch.nn.parameter.Parameter(torch.randn(h_size, h_size) * 0.05, requires_grad=True)
        self.B = torch.nn.parameter.Parameter(torch.randn(h_size, n_features)* 0.05,  requires_grad=True)
        self.C = torch.nn.parameter.Parameter(torch.randn(h_size, f_out_size)* 0.05,  requires_grad=True)
        self.D = torch.nn.parameter.Parameter(torch.randn(h_size)* 0.05,  requires_grad=True)
        
        self.E = torch.nn.parameter.Parameter(torch.randn(n_features, h_size)* 0.05, requires_grad=True)


        self.AG = Neighbor_Aggregation(n_nodes, h_size, f_out_size, edge_weights= None, device=self.device)
        
        self.H  = None
        
        self.node_idx = None
     
    
    
    def forward(self, x_in, edge_weights=None, pred_hor = 1):
        """ Forward pass of the Graph RNN
        Args:

            x_in (torch.Tensor): input tensor of shape (batch, n_time_steps, n_nodes, n_features)
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_time_steps, n_edges, 3), per edge (node1 , node2, edge_feat)
            pred_hor (int): number of time steps to predict
            """
        if x_in.shape != (x_in.shape[0], x_in.shape[1], self.n_nodes, self.n_features):
            raise ValueError(f"Input tensor shape is {x_in.shape}, expected {(x_in.shape[0], x_in.shape[1], self.n_nodes, self.n_features)}")
        
        #use the edge weights of the first time step for the rest of the prediction loop. 
        # Can be a smarter estimate, but same edges is nice.
        self.AG.calc_adj(edge_weights[:,0,:,:]) 
        
        if self.node_idx is None:
            self.node_idx = torch.zeros(self.n_nodes)
            self.node_idx = edge_weights[0, :, 0].unique() 
        self.H_prev = torch.zeros((x_in.shape[0], self.n_nodes, self.h_size), dtype=torch.float32, device=self.device)    
        self.H = self.init_H.clone().unsqueeze(0).unsqueeze(0).expand(x_in.shape[0], self.n_nodes, self.h_size)

        x_pred = []
        for i in range(x_in.shape[1] + pred_hor):
            
            if i < x_in.shape[1]:
                x_pred.append( self.forward_step(x_in[:, i, :], edge_weights=edge_weights[:,i,:,:]) )
            else:
                #use most recent edge weights
                x_pred.append( self.forward_step(x_pred[-1], edge_weights=edge_weights[:,x_in.shape[1]-1,:,:]) )
                
        x_out  = torch.stack(x_pred, dim=1).to(self.device)

        return x_out
        
    
    def forward_step(self, x_in, edge_weights=None):
        if edge_weights is None:
            if self.edge_weights is None:
                raise ValueError("Edge weights not provided. Provide edge weights to the forward pass or during initialization.")
            edge_weights = self.edge_weights
            

       
        self.neigh_ag = self.AG(self.H, node_idx=self.node_idx)
        
        self.A_expanded = self.A.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.h_size, self.h_size)
        self.B_expanded = self.B.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.h_size, self.n_features)
        self.C_expanded = self.C.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.h_size, self.f_out_size)
        self.D_expanded = self.D.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.h_size)
        self.E_expanded = self.E.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.n_features, self.h_size)
        
        

        # Perform matrix multiplications with explicit dimension specification using einsum
        AH = torch.einsum('bnij,bnj->bni', self.A_expanded, self.H_prev)
        BX = torch.einsum('bnij,bnj->bni', self.B_expanded, x_in)
        CAG = torch.einsum('bnij,bnj->bni', self.C_expanded, self.neigh_ag)

        # self.H = torch.tanh(AH + BX + CAG + self.D_expanded) 
        # Tanh saturates the gradients, so we use ReLU instead
        self.H = torch.tanh(self.H_prev + AH + BX + CAG + self.D_expanded)
        x_out = torch.einsum('bnij,bnj->bni', self.E_expanded, self.H)
        self.H_prev = self.H
        return x_out