import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
from tqdm import tqdm
import  numpy as np
from preprocessor import Preprocessor


class GraphRNN_dataset(torch.utils.data.Dataset):
    def __init__(self, epi_dates, flow_dataset = "data/daily_county2county_2019_01_01.csv", epi_dataset = "data_epi/epidemiology.csv", input_hor=4, pred_hor=1):
        super(GraphRNN_dataset, self).__init__()
        
        preprocessor = Preprocessor(flow_dataset, epi_dataset, epi_dates, plottable=True)
        kron_flow_df, signals_df = preprocessor.disjoint_manual_kronecker()
    
        self.input_hor = input_hor
        self.pred_hor = pred_hor
        
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
            # print("pop_flows: ", kron_flow_df.iloc[j]['pop_flows'])
            
        for i in range(1, self.n_time):
            self.edge_weights[i][:][:] = self.edge_weights[i-1][:][:]   
        self.edge_weights = self.edge_weights.float()
        
        
        # # Populate node_data with signals
        # for i, date in tqdm(enumerate(self.dates)):
            
        #     for j, node_id in tqdm(enumerate(self.node_ids)):
        #         condition = (signals_df['geoid_o'] == node_id) & (signals_df['date'] == date)
        #         filtered = signals_df.loc[condition, 'new_confirmed']
                
        #         if not filtered.empty:
        #             self.node_data[i][j][0] = filtered.values[0]  # Ensure single value assignment
        #         else:
        #             self.node_data[i][j][0] = 0  # Handle case where no match is found
        # Create a DataFrame to store node_data
        
        index = pd.MultiIndex.from_product([self.dates, self.node_ids], names=['date', 'geoid_o'])
        node_data_df = pd.DataFrame(0, index=index, columns=['new_confirmed'])

        # Populate node_data_df with the values from signals_df
        signals_indexed = signals_df.set_index(['date', 'geoid_o'])
        node_data_df.update(signals_indexed['new_confirmed'])

        # Convert the DataFrame to a numpy array
        node_data_array = node_data_df.unstack(level='geoid_o').values
        node_data_array = np.expand_dims(node_data_array, axis=2)

        # Assign the numpy array to self.node_data
        self.node_data = node_data_array
        self.node_data = torch.tensor(self.node_data).float()
        print(f"node_data: {self.node_data.shape}")
        print(f"edge_weights: {self.edge_weights.shape}")
        print("=====================================")
        print(self.node_data)
        print(self.edge_weights)

                
    def __len__(self):
        return self.n_time - (self.input_hor + self.pred_hor) +1
    
    def __getitem__(self, idx):
        edge_weights = self.edge_weights[idx : idx + self.input_hor + self.pred_hor ]
        node_data = self.node_data[idx : idx + self.input_hor + self.pred_hor ]
        
        input_edge_weights = edge_weights[:self.input_hor]
        input_node_data = node_data[:self.input_hor]
        target_edge_weights = edge_weights[self.input_hor:]
        target_node_data = node_data[self.input_hor:]
        
        return input_edge_weights, input_node_data, target_edge_weights, target_node_data
        
    
class GraphRNN_DataSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, input_hor, pred_hor):
        self.dataset = dataset
        self.input_hor = input_hor
        self.pred_hor = pred_hor
        self.start_idx = self.new_seq_start_idx()
        super(GraphRNN_DataSampler, self).__init__()

    def new_seq_start_idx(self):
        start_idx = torch.randint(0, self.dataset.n_time - (self.input_hor + self.pred_hor), (1,)).item()
        
        self.start_idx_list = [start_idx + (self.input_hor + self.pred_hor)*i for i in range( int(np.floor( self.dataset.n_time / (self.input_hor + self.pred_hor))))]
        return 
    
    def __len__(self):
        return  len(self.start_idx_list)
     
    def __iter__(self):
        self.new_seq_start_idx()
        return iter(self.start_idx_list)
      
      

if __name__ == '__main__':
    
    flow_dataset = "data/daily_county2county_2019_01_01.csv"
    epi_dataset = "data_epi/epidemiology.csv"
    epi_dates = ["2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"]

    input_hor = 4
    pred_hor = 2
    
    data_set = GraphRNN_dataset(epi_dates = epi_dates, flow_dataset = flow_dataset, epi_dataset = epi_dataset,  input_hor=input_hor, pred_hor=pred_hor)
    data_sampler = GraphRNN_DataSampler(data_set, input_hor=input_hor, pred_hor=pred_hor)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=3, sampler=data_sampler, num_workers=3)
    
    for input_edge_weights, input_node_data, target_edge_weights, target_node_data in data_loader:
        print(f"input_edge_weights: {input_edge_weights.shape}")
        print(f"input_node_data: {input_node_data.shape}")
        print(f"target_edge_weights: {target_edge_weights.shape}")
        print(f"target_node_data: {target_node_data.shape}")
        print("=====================================")
        print(input_edge_weights)
        print(input_node_data)
        print(target_edge_weights)
        print(f"Sparsity of target_node_data: {target_node_data.eq(0).sum().item() / target_node_data.numel()}")
        
        break

