import torch
import torch.sparse as sparse
from tqdm import tqdm

class Neighbor_Aggregation(torch.nn.Module):
    def __init__(self, n_nodes, h_size , f_out_size, fixed_edge_weights=None, device='cpu', dtype=None):  
        """ Initialize the Neighbor Aggregation
        Args:
            n_nodes (int): number of nodes in the graph
            h_size (int): size of the hidden state
            f_out_size (int): size of the output vector
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1 , node2, edge_feat)"""
            
        super(Neighbor_Aggregation, self).__init__()
        self.device = device
        self.dtype = dtype
        torch.device(self.device)
        self.n_nodes = n_nodes
        self.h_size = h_size
        self.f_out_size = f_out_size
        self.fixed_edge_weights = fixed_edge_weights
        self.H_adj = None
        self.node_idx = None
        
        self.node1_idx = None
        self.node2_idx = None
        
        self.CALCED_FIXED = False
        if self.fixed_edge_weights is not None:
            if (self.node_idx is None):
                self.node_idx = torch.cat((self.fixed_edge_weights[:, 0].unique(), self.fixed_edge_weights[:, 1].unique())).unique().tolist()
                self.node_idx.sort()
                
                self.node1_idx = torch.tensor([self.node_idx.index(n.item()) for n in self.fixed_edge_weights[:, 0]], device=self.device, dtype=torch.int)
                self.node2_idx = torch.tensor([self.node_idx.index(n.item()) for n in self.fixed_edge_weights[:, 1]], device=self.device, dtype=torch.int)
            
            self.H_adj = self.calc_adj( 1, self.fixed_edge_weights, fixed=True)
            self.CALCED_FIXED = True


    def forward (self, H,  edge_weights = None):
        """ Forward pass of the Neighbor Aggregation
        Args:
            H (torch.Tensor): hidden state tensor of shape (batch, n_nodes, h_size)
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1, node2, edge_feat)
            node_idx (list): list of node indices"""
        
        if edge_weights is not None:
            if self.fixed_edge_weights is None:
                self.H_adj = self.calc_adj(H.shape[0],  edge_weights, fixed=False)
            else:
                raise ValueError("Edge weights are constant because they were provided at initialization, pls do not provide edge weights at forward pass.")
        else:
            if self.fixed_edge_weights is None:
                raise ValueError("Edge weights are not constant and were not provided at initialization.")
            else:
                self.H_adj = self.calc_adj( H.shape[0], fixed=True )
            
        AG = torch.zeros(H.shape, dtype=self.dtype, device=self.device)
        H = H.type(torch.float32)
        
        #TODO this is bad, but I don't know how to do this in a better way
        for batch in range(H.shape[0]):
            AG[batch] = sparse.mm(self.H_adj[batch], H[batch])
        return AG

    def calc_adj(self,  batch_size, edge_weights=None, fixed = True):
        """Calculate the adjacency matrix H_adj
        Args:
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1, node2, edge_feat)
        """
        indices = []
        values = []
        
        if fixed:
            if not self.CALCED_FIXED:
                self.weight = edge_weights[:, 2].float().to(self.device)
                if edge_weights is None:
                    raise ValueError("Edge weights not provided. Provide edge weights to the forward pass or during initialization.")

                self.n_edges = edge_weights.shape[0]
                
            for batch in range(batch_size):
                batch_indices = torch.full((self.n_edges,), batch, dtype=torch.int, device=self.device)
                
                # Append to the lists
                indices.append(torch.stack([batch_indices, self.node1_idx, self.node2_idx], dim=0))
                values.append(self.weight)

            # Concatenate all batch indices and values
            indices = torch.cat(indices, dim=1)
            values = torch.cat(values)

            # Create the sparse adjacency matrix
            self.adj_matrix = torch.sparse_coo_tensor(indices, values, (batch_size, self.n_nodes, self.n_nodes), dtype=torch.float32, device=self.device)
        
            return self.adj_matrix
        else:
            if batch_size != edge_weights.shape[0]:
                raise ValueError(f"Batch size {batch_size} does not match edge weights batch size {edge_weights.shape[0]}")
            
            if self.node_idx is None:
                self.node_idx = torch.cat((edge_weights[:, :, 0].unique(), edge_weights[:, :, 1].unique())).unique().tolist()
                self.node_idx.sort()
                self.node_idx = torch.tensor(self.node_idx, dtype=torch.int, device=self.device)
                self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_idx.tolist())}
                
            batch_size = edge_weights.shape[0]
            n_edges = edge_weights.shape[1]



            for batch in range(batch_size):
                batch_indices = torch.full((n_edges,), batch, dtype=torch.int, device=self.device)
                
                batch_edges = edge_weights[batch]
                node1 = batch_edges[:, 0].int().to(self.device)
                node2 = batch_edges[:, 1].int().to(self.device)
                weight = batch_edges[:, 2].float().to(self.device)
                
                # Map node IDs to indices
                self.node1_idx = torch.tensor([self.node_id_to_idx[n.item()] for n in node1], device=self.device)
                self.node2_idx = torch.tensor([self.node_id_to_idx[n.item()] for n in node2], device=self.device)
                

                
                
                # Append to the lists
                indices.append(torch.stack([batch_indices, self.node1_idx, self.node2_idx], dim=0))
                values.append(weight)

            # Concatenate all batch indices and values
            indices = torch.cat(indices, dim=1)
            values = torch.cat(values)

            # Create the sparse adjacency matrix
            adj_matrix = torch.sparse_coo_tensor(indices, values, (batch_size, self.n_nodes, self.n_nodes), dtype=torch.float32, device=self.device)
            return adj_matrix
        
