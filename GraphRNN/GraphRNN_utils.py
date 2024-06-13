import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from preprocessor import Preprocessor
import matplotlib.pyplot as plt

class GraphRNN_dataset(torch.utils.data.Dataset):
    def __init__(self, epi_dates, flow_dataset="data/daily_county2county_2019_01_01.csv", epi_dataset="data_epi/epidemiology.csv", input_hor=4, pred_hor=1, fake_data=False):
        super(GraphRNN_dataset, self).__init__()
        self.input_hor = input_hor
        self.pred_hor = pred_hor
        
        if fake_data:
            self.n_time = 59
            self.n_nodes = 4
            self.n_edges = 15
            self.n_features = 1
            self.node_data = torch.randn((self.n_time, self.n_nodes, self.n_features), dtype=torch.float32)

            fake_idx = torch.randint(0, 10000, (self.n_nodes,))
            self.edge_weights = torch.zeros((self.n_time, self.n_edges, 3), dtype=torch.float32)

            for time in range(self.n_time):
                for node in range(self.n_edges):
                    origin = torch.randint(0, self.n_nodes, (1,)).item()
                    destination = torch.randint(0, self.n_nodes, (1,)).item()
                    if time == 0:  
                        self.edge_weights[time, node, 0] = fake_idx[origin]
                        self.edge_weights[time, node, 1] = fake_idx[destination]
                        self.edge_weights[time, node, 2] = (torch.rand((1,)).item()-0.5)
                    else:
                        self.edge_weights[time, node, 0] = fake_idx[origin]
                        self.edge_weights[time, node, 1] = fake_idx[destination]
                        self.edge_weights[time, node, 2] = self.edge_weights[time-1, node, 2] + 0.1*(torch.rand((1,)).item() - 0.5)
                for node in range(self.n_nodes):
                    if time == 0:
                        self.node_data[time, node, 0] = torch.rand((1,)).item()-0.5
                    else:
                        self.node_data[time, node, 0] = self.node_data[time-1, node, 0] + 0.1*(torch.rand((1,)).item() - 0.5)
            return
        
        preprocessor = Preprocessor(flow_dataset, epi_dataset, epi_dates, plottable=True)
        kron_flow_df, signals_df = preprocessor.disjoint_manual_kronecker()
    
        self.n_time = signals_df['date'].nunique()
        self.dates = signals_df['date'].unique().tolist()
        self.node_ids = signals_df['geoid_o'].unique().tolist()
        self.n_edges = len(kron_flow_df)
        self.n_nodes = len(signals_df['geoid_o'].unique())
        
        self.n_features = 1
        
        self.edge_weights = torch.zeros((self.n_time, self.n_edges, 3), dtype=torch.float32)
        self.node_data = torch.zeros((self.n_time, self.n_nodes, self.n_features), dtype=torch.float32)

        for j in tqdm(range(self.n_edges)):
            self.edge_weights[0][j][0] = kron_flow_df.iloc[j]['geoid_o']
            self.edge_weights[0][j][1] = kron_flow_df.iloc[j]['geoid_d']
            self.edge_weights[0][j][2] = kron_flow_df.iloc[j]['pop_flows']
            # if self.edge_weights[0][j][2] <= 1:
            #     self.edge_weights[0][j][2] = 0
                
        self.edge_weights = self.edge_weights[0].repeat(self.n_time, 1, 1)
        self.edge_weights = self.edge_weights.float()
        
        index = pd.MultiIndex.from_product([self.dates, self.node_ids], names=['date', 'geoid_o'])
        node_data_df = pd.DataFrame(0, index=index, columns=['new_confirmed'])
        signals_indexed = signals_df.set_index(['date', 'geoid_o'])
        node_data_df.update(signals_indexed['new_confirmed'])
        node_data_array = node_data_df.unstack(level='geoid_o').values
        node_data_array = np.expand_dims(node_data_array, axis=2)

        self.node_data = torch.tensor(node_data_array, dtype=torch.float32)
        print(f"node_data: {self.node_data.shape}")
        print(f"edge_weights: {self.edge_weights.shape}")
        print("=====================================")
        self.node_ids_from_edges = torch.cat((self.edge_weights[:, :, 0].unique(), self.edge_weights[:, :, 1].unique())).unique().tolist()
        self.node_ids_from_edges.sort()
        self.node_ids.sort()
        if self.node_ids != self.node_ids_from_edges:
            num_equal_elements = len(set(self.node_ids) & set(self.node_ids_from_edges))
            print(f"Number of equal elements: {num_equal_elements}")
            print(f"Node IDs from edges: {self.node_ids_from_edges[num_equal_elements -10: num_equal_elements + 10]}")
            raise ValueError(f"Node ID lists do not contain the same nodes.  Node IDs from edges: {self.node_ids_from_edges[:10]}, Node IDs from signals: {self.node_ids[:10]}. Sizes: {len(self.node_ids_from_edges)}, {len(self.node_ids)}")

    def __len__(self):
        return self.n_time - (self.input_hor + self.pred_hor) + 1
    
    def __getitem__(self, idx):
        edge_weights = self.edge_weights[idx:idx + self.input_hor + self.pred_hor]
        node_data = self.node_data[idx:idx + self.input_hor + self.pred_hor]
        
        input_edge_weights = edge_weights[:self.input_hor]
        input_node_data = node_data[:self.input_hor]
        target_edge_weights = edge_weights[self.input_hor:]
        target_node_data = node_data[self.input_hor:]
        
        if input_node_data.shape[0] != self.input_hor:
            raise ValueError(f"Input node data shape {input_node_data.shape} does not match input horizon {self.input_hor}")
        
        if target_node_data.shape[0] != self.pred_hor:
            raise ValueError(f"Target node data shape {target_node_data.shape} does not match prediction horizon {self.pred_hor}")
        
        return input_edge_weights, input_node_data, target_edge_weights, target_node_data
    def visualize(self, index):
        if index >= len(self):
            raise ValueError(f"Index {index} out of bounds for dataset of length {len(self)}")
        
        input_edge_weights, input_node_data, target_edge_weights, target_node_data = self[index]
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Plot node data
        axes[0].set_title("Node Data Over Time")
        for node in range(self.n_nodes):
            axes[0].plot(input_node_data[:, node, 0].numpy(), label=f"Node {node}")
        axes[0].legend()
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Node Feature Value")
        
        # Plot edge weights
        axes[1].set_title("Edge Weights Over Time")
        for edge in range(self.n_edges):
            axes[1].plot(input_edge_weights[:, edge, 2].numpy(), label=f"Edge {edge}")
        axes[1].legend()
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Edge Weight")

        plt.tight_layout()
        plt.show()


class GraphRNN_DataSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, input_hor, pred_hor):
        self.dataset = dataset
        self.input_hor = input_hor
        self.pred_hor = pred_hor
        self.new_seq_start_idx()
        super(GraphRNN_DataSampler, self).__init__()

    def new_seq_start_idx(self):
        start_idx = torch.randint(0, (self.input_hor + self.pred_hor), (1,)).item()
        self.start_idx_list = [start_idx + (self.input_hor + self.pred_hor) * i for i in range(int(np.floor((self.dataset.n_time - start_idx) / (self.input_hor + self.pred_hor))))]
    
    def __len__(self):
        return len(self.start_idx_list)
     
    def __iter__(self):
        self.new_seq_start_idx()
        return iter(self.start_idx_list)


if __name__ == '__main__':
    flow_dataset = "data/daily_county2county_2019_01_01.csv"
    epi_dataset = "data_epi/epidemiology.csv"
    epi_dates = ["2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"]

    input_hor = 4
    pred_hor = 2
    
    data_set = GraphRNN_dataset(epi_dates=epi_dates, 
                                flow_dataset=flow_dataset,
                                epi_dataset=epi_dataset,
                                input_hor=input_hor,
                                pred_hor=pred_hor,
                                fake_data=False)

    # data_sampler = GraphRNN_DataSampler(data_set, input_hor=input_hor, pred_hor=pred_hor)
    # data_loader = torch.utils.data.DataLoader(data_set, batch_size=3, sampler=data_sampler, num_workers=3)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=3, num_workers=3)

    
    for i in range(10):
        print(f"Epoch: {i}")
        for input_edge_weights, input_node_data, target_edge_weights, target_node_data in data_loader:
            print(f"input_edge_weights: {input_edge_weights.shape}")
            print(f"input_node_data: {input_node_data.shape}")
            print(f"target_edge_weights: {target_edge_weights.shape}")
            print(f"target_node_data: {target_node_data.shape}")
            print("=====================================")
            print(f"Sparsity of target_node_data: {target_node_data.eq(0).sum().item() / target_node_data.numel()}")
        
       
