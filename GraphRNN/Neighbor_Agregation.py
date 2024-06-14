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
        self.node_idx = None
        
        if self.edge_weights is not None:
            self.CONST_EDGE_WEIGHT = True
            self.node_idx = torch.cat((edge_weights[:, :, 0].unique(), edge_weights[:, :, 1].unique())).unique().tolist().sort()
            self.H_adj = self.calc_H_adj(edge_weights)
        else:
            self.CONST_EDGE_WEIGHT = False
        
    def forward (self, H,  edge_weights = None, node_idx = None):
        """ Forward pass of the Neighbor Aggregation
        Args:
            H (torch.Tensor): hidden state tensor of shape (batch, n_nodes, h_size)
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1, node2, edge_feat)
            node_idx (list): list of node indices"""
        if self.node_idx is None:
            self.node_idx = torch.cat((edge_weights[:, :, 0].unique(), edge_weights[:, :, 1].unique())).unique().tolist()
            self.node_idx.sort()
            self.node_idx = torch.tensor(self.node_idx, dtype=torch.int, device=self.device)
        
        AG = torch.zeros(H.shape, dtype=torch.float32, device=self.device)
        for batch in range(H.shape[0]):
            AG[batch] = sparse.mm(self.adj_matrix[batch], H[batch])
        return AG

    def calc_adj(self, edge_weights):
        """Calculate the adjacency matrix H_adj
        Args:
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1, node2, edge_feat)
        """
        if self.node_idx is None:
            self.node_idx = torch.cat((edge_weights[:, :, 0].unique(), edge_weights[:, :, 1].unique())).unique().tolist()
            self.node_idx.sort()
            self.node_idx = torch.tensor(self.node_idx, dtype=torch.int, device=self.device)
            self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_idx.tolist())}
            
        batch_size = edge_weights.shape[0]
        n_edges = edge_weights.shape[1]

        indices = []
        values = []

        for batch in range(batch_size):
            batch_edges = edge_weights[batch]
            
            node1 = batch_edges[:, 0].int().to(self.device)
            node2 = batch_edges[:, 1].int().to(self.device)
            weight = batch_edges[:, 2].float().to(self.device)
            
            # Map node IDs to indices
            node1_idx = torch.tensor([self.node_id_to_idx[n.item()] for n in node1], device=self.device)
            node2_idx = torch.tensor([self.node_id_to_idx[n.item()] for n in node2], device=self.device)

            batch_indices = torch.full((n_edges,), batch, dtype=torch.long, device=self.device)
            
            # Append to the lists
            indices.append(torch.stack([batch_indices, node1_idx, node2_idx], dim=0))
            values.append(weight)

        # Concatenate all batch indices and values
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values)

        # Create the sparse adjacency matrix
        self.adj_matrix = torch.sparse_coo_tensor(indices, values, (batch_size, self.n_nodes, self.n_nodes), dtype=torch.float32, device=self.device)
        return 1
        
