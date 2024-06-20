import torch
import torch.sparse as sparse
from Neighbor_Agregation import Neighbor_Aggregation
from tqdm import tqdm
class Graph_RNN(torch.nn.Module):
    def __init__(self, n_nodes, n_features, h_size, f_out_size, fixed_edge_weights=None , device='cpu', dtype=torch.float32, use_neighbors=True):
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
        self.dtype = dtype
        if dtype != torch.float32:
            raise ValueError("Only float32 is supported")
        self.use_neighbors = use_neighbors
        self.fixed_edge_weights = fixed_edge_weights
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.h_size = h_size
        self.f_out_size = f_out_size
        print(f"n_nodes: {n_nodes}, n_features: {n_features}, h_size: {h_size}, f_out_size: {f_out_size}")
        
        self.init_mag = 0.01
        self.init_H = torch.nn.parameter.Parameter(torch.randn(h_size, device=self.device, dtype=self.dtype)* self.init_mag, requires_grad=True)
        
        self.A = torch.nn.parameter.Parameter(torch.randn(h_size, h_size, device=self.device, dtype=self.dtype)* self.init_mag ,  requires_grad=True)
        self.B = torch.nn.parameter.Parameter(torch.randn(h_size, n_features , device=self.device, dtype=self.dtype)* self.init_mag ,  requires_grad=True)
        self.C = torch.nn.parameter.Parameter(torch.randn(h_size, f_out_size, device=self.device, dtype=self.dtype)* self.init_mag ,  requires_grad=True)
        self.D = torch.nn.parameter.Parameter(torch.randn(h_size, device=self.device, dtype=self.dtype)* self.init_mag ,  requires_grad=True)
        
        # self.E = torch.nn.parameter.Parameter(torch.randn(h_size, h_size, device=self.device, dtype=self.dtype)* self.init_mag ,  requires_grad=True)
        # self.E2 = torch.nn.parameter.Parameter(torch.randn(n_features, h_size, device=self.device, dtype=self.dtype)* self.init_mag ,  requires_grad=True)
        #init as eye. Network will start off predicting the input
        self.F = torch.nn.parameter.Parameter(torch.eye(n_features, device=self.device, dtype=self.dtype) ,  requires_grad=True)
        
        self.G = torch.nn.parameter.Parameter(torch.randn(n_features, device=self.device, dtype=self.dtype)* self.init_mag ,  requires_grad=True)
        torch.nn.init.xavier_normal_(self.A)
        torch.nn.init.xavier_normal_(self.B)
        torch.nn.init.xavier_normal_(self.C)
        
        #test not init F at eye
        torch.nn.init.xavier_normal_(self.F)
    
        # torch.nn.init.xavier_normal_(self.E) 
        # torch.nn.init.xavier_normal_(self.E2)

        self.H2X_out_MLP = torch.nn.Sequential(
            torch.nn.Linear(h_size, h_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h_size, h_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h_size, n_features)
        )
        if self.fixed_edge_weights is not None:
            self.node_idx = self.fixed_edge_weights[:, 0].unique()
        else:
            self.node_idx = None
            
        self.AG = Neighbor_Aggregation(n_nodes, h_size, f_out_size, fixed_edge_weights= fixed_edge_weights, device=self.device, dtype=self.dtype)
        
        self.H  = None
        
        
     
    
    
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
        
        if self.node_idx is None:
            if edge_weights is None:
                raise ValueError("Edge weights not provided. Provide edge weights to the forward pass or during initialization.")
            self.node_idx = torch.zeros(self.n_nodes)
            self.node_idx = edge_weights[0, :, 0].unique() 
            
        self.H_prev = self.init_H.clone().unsqueeze(0).unsqueeze(0).expand(x_in.shape[0], self.n_nodes, self.h_size)
        self.H = self.init_H.clone().unsqueeze(0).unsqueeze(0).expand(x_in.shape[0], self.n_nodes, self.h_size)

        x_pred = []
        for i in range(x_in.shape[1] + pred_hor):
            
            if i < x_in.shape[1]:
                if edge_weights is not None:
                    x_pred.append( self.forward_step(x_in[:, i, :], edge_weights=edge_weights[:,i,:,:]) )
                else:
                    x_pred.append( self.forward_step(x_in[:, i, :]) )
            else:
                #use most recent edge weights
                if edge_weights is not None:
                    x_pred.append( self.forward_step(x_pred[-1], edge_weights=edge_weights[:,x_in.shape[1]-1,:,:]) )
                else:
                    x_pred.append( self.forward_step(x_pred[-1]) )                
        x_out  = torch.stack(x_pred, dim=1).to(self.device)

        return x_out
        
    
    def forward_step(self, x_in, edge_weights=None):
        batch_size = x_in.shape[0]
        if edge_weights is None:
            if self.fixed_edge_weights is None:
                raise ValueError("Edge weights not provided. Provide edge weights to the forward pass or during initialization.")
        if self.use_neighbors:    
            self.neigh_ag = self.AG(self.H, edge_weights=edge_weights)
        else:
            self.neigh_ag = torch.zeros(self.H.shape, dtype=self.dtype, device=self.device)
        
        self.A_expanded = self.A.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.h_size, self.h_size)
        self.B_expanded = self.B.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.h_size, self.n_features)
        self.C_expanded = self.C.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.h_size, self.f_out_size)
        self.D_expanded = self.D.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.h_size)
        # self.E_expanded = self.E.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.h_size, self.h_size)
        # self.E2_expanded = self.E2.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.n_features, self.h_size)
        self.F_expanded = self.F.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.n_features, self.n_features)
        self.G_expanded = self.G.unsqueeze(0).unsqueeze(1).expand(x_in.shape[0], self.n_nodes, self.n_features)
        # Perform matrix multiplications with explicit dimension specification using einsum
        
        AH = torch.einsum('bnij,bnj->bni', self.A_expanded, self.H_prev)
        BX = torch.einsum('bnij,bnj->bni', self.B_expanded, x_in)
        CAG = torch.einsum('bnij,bnj->bni', self.C_expanded, self.neigh_ag)
        

        
        FX = torch.einsum('bnij,bnj->bni', self.F_expanded, x_in)
        
        # print(f"self.H_prev[0,0,:5]: {self.H_prev[0,0,:self.n_features]}")
        # print(f"BX[0,0,:5]: {BX[0,0,:5]}")
        # print(f"CAG[0,0,:5]: {CAG[0,0,:5]}")
        # print(f"X_in[0,0,:]: {x_in[0,0,:]}")
        # print(f"FX[0,0,:]: {FX[0,0,:]}")
        
        # self.H = torch.tanh(AH + BX + CAG + self.D_expanded) 
        # Tanh saturates the gradients, so we use ReLU instead
        self.H = self.H_prev + torch.tanh( AH + BX + CAG + self.D_expanded)
        self.H_out= self.H2X_out_MLP(self.H.view(batch_size*self.n_nodes, self.h_size)).view(batch_size, self.n_nodes, self.n_features)
        x_out = self.H_out + FX + self.G_expanded
        self.H_prev = self.H
        return x_out