import torch
def calc_adj(self,   edge_weights=None, fixed = True):
        """Calculate the adjacency matrix H_adj
        Args:
            edge_weights (torch.Tensor): edge weights tensor of shape (batch, n_edges, 3), per edge (node1, node2, edge_feat)
        """
        indices = []
        values = []
        
        if not self.CALCED_FIXED:
            self.weight = edge_weights[:, 2].float().to(self.device)
            if edge_weights is None:
                raise ValueError("Edge weights not provided. Provide edge weights to the forward pass or during initialization.")

            self.n_edges = edge_weights.shape[0]
            
            
            # Append to the lists
            indices.append(torch.stack([self.node1_idx, self.node2_idx], dim=0))
            values.append(self.weight)

        # Concatenate all batch indices and values
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values)

        # Create the sparse adjacency matrix
        self.adj_matrix = torch.sparse_coo_tensor(indices, values, ( self.n_nodes, self.n_nodes), dtype=torch.float32, device=self.device)
    
        return self.adj_matrix
        