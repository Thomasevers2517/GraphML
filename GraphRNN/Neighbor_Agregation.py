#to do, make the matrix multiplications sparse.
import torch
import torch.sparse as sparse
from tqdm import tqdm

dropout=0.2

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
            self.H_adj = self.calc_adj(edge_weights)
        else:
            self.CONST_EDGE_WEIGHT = False
        

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



        for batch in tqdm(range(batch_size), desc="Calculating adjacency matrices", leave=False):
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
        
        print(f"Adjacency matrix shape: {self.adj_matrix.shape}")
        return 1
        
class Neighbor_Aggregation_Simple(Neighbor_Aggregation):
    def __init__(self, n_nodes, h_size , f_out_size, edge_weights=None, device='cpu'):
        super(Neighbor_Aggregation_Simple, self).__init__(n_nodes, h_size , f_out_size, edge_weights, device)
        
        
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

class Head_Graph(nn.Module):
    """one head of graph attention (not sure it is the same as the actual graph attention)
    Essentially same as the original attention head, except the mask hides non-neighbours
    Args:
        h_size (int): size of the hidden state
        f_out_size (int): size of the output vector
        adj_matrix (np.array): adjacency matrix of the graph, size (B, N, N)
    """
    def __init__(self, h_size, f_out_size,adj_matrix):
        super().__init__()
        self.query = nn.Linear(h_size,f_out_size,bias=False)
        self.key = nn.Linear(h_size,f_out_size,bias=False)
        self.value = nn.Linear(h_size,f_out_size,bias=False)
        self.register_buffer('adj_matrix', adj_matrix.float())
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape # (B, N, h_size)
        k = self.key(x) # (B, N, F_out)
        q = self.query(x) # (B, N, F_out)
        v = self.value(x) # (B, N, F_out)


        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, N, N)
        #none of the above matrices are sparse, so processing as dense paralelly in batches.
        #TO DO: FIND A WAY TO AVOID MAKING IT DENSE AGAIN
        wei = wei.masked_fill(self.adj_matrix.to_dense() == 0, float("-inf")) # mask out non-neighbours
        wei = F.softmax(wei, dim=-1) # still (B, N, N)
        wei.to_sparse() #wei is now sparse after softmax with adjacency mask
        wei = wei+ self.adj_matrix.to_sparse() # add adjacency matrix to the weights (edge informed attention)
        wei = self.dropout(wei) 
        
        out = torch.zeros((B, N, v.shape[-1]), device=x.device)
        #process batch-wise
        print(wei.shape[0])
        for b in range(wei.shape[0]):
            out[b]= torch.sparse.mm(wei[b], v[b]) # (N, N) @ (N, F_out) = (N, F_out)

        
        #out = wei @ v # (B,N,N) @ (B,N,F_out) = (B,N,F_out)

        
        return out

class MultiHeadNeighbourAttention(nn.Module):
    def __init__(self, n_heads, h_size, f_out_size, adj_matrix):
        super().__init__()
        self.heads = nn.ModuleList([Head_Graph(h_size, f_out_size, adj_matrix) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads*f_out_size, f_out_size)
        self.layer_norm = nn.LayerNorm(f_out_size)
        
    def forward(self, x):
        return self.layer_norm(self.linear(torch.cat([head(x) for head in self.heads], dim=-1))) # (B, N, F_out)
class FeedFoward(nn.Module):
  

    def __init__(self, h_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( h_size, 4 *  h_size),
            nn.ReLU(),
            nn.Linear(4 *  h_size,  h_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class BlockNeighbour(nn.Module):
    def __init__(self,n_head, h_size, f_out_size, adj):
        
        super().__init__()
       
        self.sa = MultiHeadNeighbourAttention(n_head, h_size,f_out_size,adj)
        self.ffwd = FeedFoward(h_size)
        self.ln1 = nn.LayerNorm(h_size)
        self.ln2 = nn.LayerNorm(h_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Neighbor_Attention_block(Neighbor_Aggregation):
    def __init__(self,n_layers,n_head,n_nodes, h_size, f_out_size, edge_weights=None, device='cpu'):
        super().__init__(n_nodes, h_size, f_out_size, edge_weights, device)
        self.n_layers=n_layers
        self.n_head=n_head
        
       
        self.blocks =  nn.Sequential(*[BlockNeighbour(n_head=self.n_head,h_size=self.h_size,f_out_size=self.f_out_size,adj=self.adj_matrix) for _ in range(n_layers)])
        self.ln= nn.LayerNorm(f_out_size)

    def forward(self,H,edge_weights=None,node_idx=None):
        
        H = self.blocks(H)
        H = self.ln(H)
        return H
    
class Neighbor_Attention_multiHead(Neighbor_Aggregation):
    def __init__(self,n_head,n_nodes, h_size, f_out_size, edge_weights=None, device='cpu'):
        super().__init__(n_nodes, h_size, f_out_size, edge_weights, device)
        self.sa = MultiHeadNeighbourAttention(n_head, h_size,f_out_size,self.adj_matrix)

    def forward(self,H,edge_weights=None,node_idx=None):
        
        H = self.sa(H)
        
        return H