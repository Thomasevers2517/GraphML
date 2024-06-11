import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import var
import pandas as pd

class GraphRNN_dataset(torch.utils.data.Dataset):
    def __init__(self, edge_data, node_data):
        
        self.n_time = len(edge_data)
        self.n_edges = len(edge_data[0])
        
        self.edge_weights = torch.zeros(self.n_time, self.n_edges, 3)
        
        for j in range(self.n_edges):
            self.edge_weights[0][j][0] = edge_data[0].iloc[j]['geoid_o']
            self.edge_weights[0][j][1] = edge_data[0].iloc[j]['geoid_d']
            self.edge_weights[0][j][2] = edge_data[0].iloc[j]['pop_flows']
        for i in range(1, self.n_time):
            self.edge_weights[i][:][:] = self.edge_weights[i-1][:][:]
            
                
        self.edge_weights = self.edge_weights.type(torch.LongTensor)
        
        self.node_data = torch.randn(self.n_time, self.n_nodes, 1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class GraphRNN_DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        super(GraphRNN_DataLoader, self).__init__(dataset, batch_size, shuffle, num_workers)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]    

edge_data = var.load_data()
smallsubset_nodes, small_edge_data = var.create_small_dataset(edge_data)

print("length of edge_data:", len(small_edge_data))
print(small_edge_data[0].head())



data_set = GraphRNN_dataset(small_edge_data, None)


