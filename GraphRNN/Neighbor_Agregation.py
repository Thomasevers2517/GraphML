import torch
import torch.sparse as sparse

class Neighbor_Aggregation(torch.nn.Module):
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
        self.node_idx = torch.zeros(n_nodes)
        
    def forward(self, H, edge_weights=None, node_idx=None):
        """ Forward pass of the neighbor aggregation function
        Args:
            H (torch.Tensor): hidden state tensor of shape (batch, n_nodes, h_size, h_size)
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1 , node2, edge_feat)
            """
        self.node_idx = node_idx 
        if not self.CONST_EDGE_WEIGHT:
            if edge_weights is None:
                raise ValueError("Edge weights not provided. Provide edge weights to the forward pass or during initialization.")
            else:
                self.H_adj = self.calc_H_adj(edge_weights)
                
        neigh_aggr = sparse.mm(self.H_adj, H)
        
        return neigh_aggr
    
    
    def calc_H_adj(self, edge_weights):
        """ Calculate the adjacency matrix H_adj
        Args:
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1 , node2, edge_feat)
            """      
        adj_matrix = torch.zeros((edge_weights.shape[0], self.n_nodes, self.n_nodes))
        for i in range(edge_weights.shape[0]):
            for j in range(edge_weights.shape[1]):
                node1 = int(edge_weights[i, j, 0])
                node2 = int(edge_weights[i, j, 1])
                weight = edge_weights[i, j, 2]
                node1_idx = torch.where(self.node_idx == node1)[0]
                node2_idx = torch.where(self.node_idx == node2)[0]
                adj_matrix[i, node1_idx, node2_idx] = weight
            H_adj = torch.kron(torch.eye(self.h_size), adj_matrix)
            
        if H_adj.shape != (self.h_size * self.n_nodes, self.h_size * self.n_nodes):
            raise ValueError(f"The shape of the adjacency matrix is incorrect. {H_adj.shape} != {(self.h_size * self.n_nodes, self.h_size * self.n_nodes)}")
        return H_adj
    