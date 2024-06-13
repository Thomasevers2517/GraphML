import torch
import torch.sparse as sparse
from tqdm import tqdm

class Neighbor_Aggregation(torch.nn.Module):
    def __init__(self, n_nodes, h_size , f_out_size, edge_weights=None, device='cpu'):  
        """ Initialize the Neighbor Aggregation
        Args:
            n_nodes (int): number of nodes in the graph
            h_size (int): size of the hidden state
            f_out_size (int): size of the output vector
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1 , node2, edge_feat)"""
            
        super(Neighbor_Aggregation, self).__init__()
        self.device = device
        torch.device(self.device)
        self.n_nodes = n_nodes
        self.h_size = h_size
        self.f_out_size = f_out_size
        self.edge_weights = edge_weights
        self.H_adj = None
        if self.edge_weights is not None:
            self.CONST_EDGE_WEIGHT = True
            self.H_adj = self.calc_H_adj(edge_weights)
        else:
            self.CONST_EDGE_WEIGHT = False
        self.node_idx = torch.cat((edge_weights[:, :, 0].unique(), edge_weights[:, :, 1].unique())).unique().tolist()
        
    def forward (self, H,  edge_weights = None, node_idx = None):
        adj_matrix = self.calc_adj(edge_weights)
        print(f"adj_matrix: {adj_matrix.shape}")
        print(f"H: {H.shape}")
        AG = torch.zeros(H.shape, dtype=torch.float32, device=self.device)
        for batch in range(H.shape[0]):
            AG[batch] = sparse.mm(adj_matrix[batch], H[batch])
        return AG

    def calc_adj(self, edge_weights):
        """Calculate the adjacency matrix H_adj
        Args:
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1, node2, edge_feat)
        """
        if len(self.node_idx) != self.n_nodes:
            raise ValueError(f"Number of nodes in edge_weights ({len(self.node_idx)}) is different from n_nodes ({self.n_nodes})")
        batch_size = edge_weights.shape[0]
        n_edges = edge_weights.shape[1]

        indices = []
        values = []

        for batch in range(batch_size):
            for i in range(n_edges):
                node1 = edge_weights[batch, i, 0].int().to(self.device)
                node2 = edge_weights[batch, i, 1].int().to(self.device)
                weight = edge_weights[batch, i, 2].float().to(self.device)
                node1_idx = torch.where(self.node_idx == node1)[0].item()
                node2_idx = torch.where(self.node_idx == node2)[0].item()

                indices.append([batch, node1_idx, node2_idx])
                values.append(weight)

        indices = torch.tensor(indices, dtype=torch.long).t()
        values = torch.tensor(values, dtype=torch.float32)

        adj_matrix = torch.sparse_coo_tensor(indices, values, (batch_size, self.n_nodes, self.n_nodes), dtype=torch.float32, device=self.device)

        return adj_matrix
        
