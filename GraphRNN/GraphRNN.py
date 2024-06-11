import torch
import torch.sparse as sparse

class Neighbor_Aggregation(torch.nn.module):
    def __init__(self, n_nodes, h_size , f_out_size, edge_weights=None):
       
        super(Neighbor_Aggregation, self).__init__()
        self.n_nodes = n_nodes
        self.h_size = h_size
        self.f_out_size = f_out_size
        self.edge_weights = edge_weights
        if self.edge_weights is not None:
            self.CONST_EDGE_WEIGHT = True
            self.H_adj = self.calc_H_adj(edge_weights)
        else:
            self.CONST_EDGE_WEIGHT = False
        
    def forward(self, H, edge_weights=None):
        """ Forward pass of the neighbor aggregation function
        Args:
            H (torch.Tensor): hidden state tensor of shape (batch, n_nodes, h_size, h_size)
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1 , node2, edge_feat)
            """
            
        if not self.CONST_EDGE_WEIGHT:
            if edge_weights is None:
                raise ValueError("Edge weights not provided. Provide edge weights to the forward pass or during initialization.")
            else:
                edge_weights = self.edge_weights
                self.H_adj = self.calc_H_adj(edge_weights)
                
        neigh_aggr = sparse.mm(self.H_adj, H)
        
        return neigh_aggr
    
    
    def calc_H_adj(self, edge_weights):
        """ Calculate the adjacency matrix H_adj
        Args:
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1 , node2, edge_feat)
            """      
        adj_matrix = torch.zeros(edge_weights.shape[0], self.n_nodes, self.n_nodes)
        for i in range(edge_weights.shape[0]):
            for j in range(edge_weights.shape[1]):
                node1 = int(edge_weights[i, j, 0])
                node2 = int(edge_weights[i, j, 1])
                weight = edge_weights[i, j, 2]
                adj_matrix[i, node1, node2] = weight
            H_adj = torch.kron(torch.eye(self.h_size), adj_matrix)
            
        if H_adj.shape != (self.h_size * self.n_nodes, self.h_size * self.n_nodes):
            raise ValueError(f"The shape of the adjacency matrix is incorrect. {H_adj.shape} != {(self.h_size * self.n_nodes, self.h_size * self.n_nodes)}")
        return H_adj
    
class Node_RNN(torch.nn.module):
    def __init__(self, node_id, n_features, h_size, f_out_size, RNN_trans_mats, AG, edge_weights=None):
        """ Initialize the RNN
        Args:
            node_id (int): node id of the node
            n_features (int): number of features in the input tensor
            h_size (int): size of the hidden state
            f_size (int): size of the vector returned by the neighbor aggregation function
            """
        super(Node_RNN, self).__init__()
        
        self.node_id = node_id
        self.h_size = h_size
        self.n_features = n_features
        self.f_out_size = f_out_size    
        
        self.edge_weights = edge_weights
        self.H_0 = torch.zeros(h_size, h_size)
        self.H   = torch.zeros(h_size, h_size)
        
        # Define the weight matrices
        #H t+1 = tanh(AH t + B X_in t + C F(neighbor hidden states))
        self.A = RNN_trans_mats["A"]
        self.B = RNN_trans_mats["B"]
        self.C = RNN_trans_mats["C"]
        
        self.D = RNN_trans_mats["D"]
        self.E = RNN_trans_mats["E"]
        
        # Define the neighbor aggregation function
        self.AG  = AG
        
    def forward(self, x_in, hidden_state, edge_weights=None):
        """ Forward pass of the RNN
        Args:
            x_in (torch.Tensor): input tensor of shape (batch, seq_size, n_features)
            hidden_state (torch.Tensor): hidden state tensor of shape (batch, h_size, h_size)
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edge_feat)

            """
        if (edge_weights is None and self.edge_weights is None):
            raise ValueError("Edge weights not provided. Provide edge weights to the forward pass or during initialization.")
        
        self.H = torch.tanh(torch.matmul(self.A, hidden_state) + torch.matmul(self.B, x_in) + torch.matmul(self.C, self.AG(hidden_state, edge_weights)) + self.E)
        x_out = torch.matmul(self.D, self.H)
        return x_out
    
class Graph_RNN(torch.nn.module):
    def __init__(self, n_nodes, n_features, h_size, f_out_size, edge_weights):
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
        
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.h_size = h_size
        self.f_out_size = f_out_size
        
        self.A = torch.nn.parameter.Parameter(torch.randn(h_size, h_size)) 
        self.B = torch.nn.parameter.Parameter(torch.randn(h_size, n_features))
        self.C = torch.nn.parameter.Parameter(torch.randn(h_size, f_out_size))
        
        self.D = torch.nn.parameter.Parameter(torch.randn(n_features, h_size))
        self.E = torch.nn.parameter.Parameter(torch.randn(h_size, h_size))
        

        self.AG = Neighbor_Aggregation(n_nodes, h_size, f_out_size, edge_weights)
        
        self.H  = None
    
    
    def  forward(self, x_in, edge_weights=None):
   
        if self.H is None:
            self.H = torch.zeros(x_in.shape[0], self.n_nodes, self.h_size, self.h_size)
       
        self.neigh_ag = self.AG(self.H, edge_weights)
        
        self.H = torch.tanh(torch.matmul(self.A, self.H) + torch.matmul(self.B, x_in) + torch.matmul(self.C, self.neigh_ag) + self.D)    
        
        x_out = torch.matmul(self.E, self.H)
    
    
        
    
