import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
from tqdm import tqdm
from GraphRNN_utils import GraphRNN_dataset, GraphRNN_DataSampler
from GraphRNN import Graph_RNN, Neighbor_Aggregation
import matplotlib.pyplot as plt
if __name__ == "__main__":
    flow_dataset = "data/daily_county2county_2019_01_01.csv"
    epi_dataset = "data_epi/epidemiology.csv"
    epi_dates = ["2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12",
                 "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16",
                
                 


                 ]


    input_hor = 4
    pred_hor = 1



    data_set = GraphRNN_dataset(epi_dates = epi_dates,
                                flow_dataset = flow_dataset,
                                epi_dataset = epi_dataset,
                                input_hor=input_hor,
                                pred_hor=pred_hor,
                                fake_data=False)
    # data_set.visualize(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    data_sampler = GraphRNN_DataSampler(data_set, input_hor=input_hor, pred_hor=pred_hor)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, sampler=data_sampler, num_workers=3)
    

    model  = Graph_RNN(n_nodes = data_set.n_nodes, n_features = data_set.n_features, h_size = 10, f_out_size =10, device=device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0010)

    torch.autograd.set_detect_anomaly(False)

    model.to(device)

    losses = []
    for epoch in range(30):
        print(f"Epoch: {epoch}")
        print("Loading Data")
        for input_edge_weights, input_node_data, target_edge_weights, target_node_data in data_loader:
            print("Moving data to device")
            input_edge_weights = input_edge_weights.to(device)
            input_node_data = input_node_data.to(device)
            target_edge_weights = target_edge_weights.to(device)
            target_node_data = target_node_data.to(device)
            print("Data loaded to device")
            
            print("Forward pass")
            output = model(x_in=input_node_data, edge_weights = input_edge_weights, pred_hor = pred_hor)
            
            print(f"output: {output.shape}")
            print(f"target_node_data: {target_node_data.shape}")
            
            
            # print(f"output: {output}")
            # print(f"target_node_data: {target_node_data}")
              
            loss = criterion(output[:,-pred_hor:,:,:], target_node_data[:,:pred_hor,:,:])
            loss += 0.01 * criterion(output[:,:-1,:,:], input_node_data[:,:,:])
            optimizer.zero_grad()
            
            print("Backward pass")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(f"loss: {loss}")
            
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    
    plt.show()

    


