import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import pandas as pd
from tqdm import tqdm
from GraphRNN_utils import GraphRNN_dataset, GraphRNN_DataSampler
from GraphRNN import Graph_RNN, Neighbor_Aggregation
import matplotlib.pyplot as plt
import json
import torch.profiler

def train(model, data_loader, criterion, optimizer, pred_hor, device, n_epochs =10, save_name=None, max_grad_norm=1):
    losses = []
    parameter_mag = {param_name: [] for param_name, param in model.named_parameters()}
    gradients = {}
    gradients["pre_limit"] = {param_name: [] for param_name, param in model.named_parameters()}
    gradients["post_limit"] = {param_name: [] for param_name, param in model.named_parameters()}
    hidden_states = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        batch_num = 0
        for input_edge_weights, input_node_data, target_edge_weights, target_node_data in tqdm(data_loader):
            if batch_num==1 and epoch == 1:
                # prof.step()
                pass

            
            input_hor = input_node_data.shape[1]
            input_edge_weights = input_edge_weights.to(device)
            input_node_data = input_node_data.to(device)
            target_edge_weights = target_edge_weights.to(device)
            target_node_data = target_node_data.to(device)
            # output = model(x_in=input_node_data, edge_weights = input_edge_weights, pred_hor = pred_hor)
            output = model(x_in=input_node_data, pred_hor = pred_hor)

            
            # print(f"output: {output}")
            # print(f"target_node_data: {target_node_data}")
              
            loss = criterion(output[:,-pred_hor:,:,:], target_node_data[:,:pred_hor,:,:])
            loss += 0 * criterion(output[:,:input_hor,:,:], input_node_data[:,:,:])
            
            optimizer.zero_grad()
            loss.backward()
            for param_name, param in model.named_parameters():
                try:
                    parameter_mag[param_name].append(param.abs().mean().item())
                    gradients["pre_limit"][param_name].append(param.grad.norm().item())
                except Exception as e:
                    parameter_mag[param_name].append(param.abs().mean().item())
                    gradients["pre_limit"][param_name].append(0)
                     
            hidden_state_mag = model.H.abs().mean().item()
            hidden_states.append(hidden_state_mag)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            for param_name, param in model.named_parameters():
                try:
                    gradients["post_limit"][param_name].append(param.grad.norm().item())
                except:
                    gradients["post_limit"][param_name].append(0)

            optimizer.step()
            epoch_loss += loss.item()
            batch_num += 1
        
        #Calculate the average loss per prediction, so per node, per time step
        epoch_loss = epoch_loss/((pred_hor) * len(data_loader) * input_node_data.shape[1])
        losses.append(epoch_loss)
        print(f"EPOCH: {epoch} ", end="")
        
        print(f"$ Loss: { epoch_loss:.3e} ")
        
        print(f"Input: {input_node_data[0, :, :5, 0]} ")
        print(f"$ Output: {output[0, -pred_hor:, :5, 0]}")
        print(f"$ Target: {target_node_data[0, :, :5, 0]}")
    if save_name is not None:
        save_string = f"{save_name}_epoch_{epoch}_lr_{optimizer.param_groups[0]['lr']}_batch_{data_loader.batch_size} _pred_{pred_hor}_input_{input_hor}_num_dates_{len(data_loader)}_h_size_{model.h_size}"
        torch.save(model.state_dict(), f"models\model_state_dict_{save_string}.pth")
        with open(f"losses\losses_{save_string}.json", "w") as f:
            json.dump(losses, f)
    return losses, parameter_mag, gradients, hidden_states

config = {  "n_epochs": 800,
            "num_dates": 9,
            "input_hor": 7,
            "pred_hor": 1,
            "h_size": 170,
            "batch_size": 5,
            "lr": 0.0003,
            "use_neighbors": True,
            "max_grad_norm": 1,
            
         }



if __name__ == "__main__":
    print("Starting training run...")
    flow_dataset = "data/daily_county2county_2019_01_01.csv"
    epi_dataset = "data_epi/epidemiology.csv"
    epi_dates = ["2020-06-09", "2020-06-10", "2020-06-11", "2020-06-12", "2020-06-12", "2020-06-13", "2020-06-14", "2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18", "2020-06-19", "2020-06-20", "2020-06-21", "2020-06-22", "2020-06-23", "2020-06-24", "2020-06-25", "2020-06-26", "2020-06-27", "2020-06-28", "2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02", "2020-07-03", "2020-07-04", "2020-07-05", "2020-07-06", "2020-07-07", "2020-07-08", "2020-07-09", "2020-07-10", "2020-07-11", "2020-07-12", "2020-07-13", "2020-07-14", "2020-07-15", "2020-07-16", "2020-07-17", "2020-07-18", "2020-07-19", "2020-07-20", "2020-07-21", "2020-07-22", "2020-07-23", "2020-07-24", "2020-07-25", "2020-07-26", "2020-07-27", "2020-07-28", "2020-07-29", "2020-07-30", "2020-07-31", "2020-08-01", "2020-08-02", "2020-08-03", "2020-08-04", "2020-08-05", "2020-08-06", "2020-08-07", "2020-08-08", "2020-08-09", "2020-08-10", "2020-08-11", "2020-08-12", "2020-08-13", "2020-08-14", "2020-08-15", "2020-08-16", "2020-08-17",
                    "2020-08-18", "2020-08-19", "2020-08-20", "2020-08-21", "2020-08-22", "2020-08-23", "2020-08-24", "2020-08-25", "2020-08-26", "2020-08-27", "2020-08-28", "2020-08-29", "2020-08-30", "2020-08-31", "2020-09-01", "2020-09-02", "2020-09-03", "2020-09-04", "2020-09-05", "2020-09-06", "2020-09-07", "2020-09-08", "2020-09-09", "2020-09-10", "2020-09-11", "2020-09-12", "2020-09-13", "2020-09-14", "2020-09-15", "2020-09-16", "2020-09-17", "2020-09-18", "2020-09-19", "2020-09-20", "2020-09-21", "2020-09-22", "2020-09-23", "2020-09-24", "2020-09-25", "2020-09-26", "2020-09-27", "2020-09-28", "2020-09-29", "2020-09-30", "2020-10-01", "2020-10-02", "2020-10-03", "2020-10-04", "2020-10-05", "2020-10-06", "2020-10-07", "2020-10-08", "2020-10-09", "2020-10-10", "2020-10-11", "2020-10-12", "2020-10-13", "2020-10-14", "2020-10-15", "2020-10-16", "2020-10-17", "2020-10-18", 
                    "2020-10-19", "2020-10-20", "2020-10-21", "2020-10-22", "2020-10-23", "2020-10-24", "2020-10-25", "2020-10-26", "2020-10-27", "2020-10-28", "2020-10-29", "2020-10-30", "2020-10-31", "2020-11-01", "2020-11-02", "2020-11-03", "2020-11-04", "2020-11-05", "2020-11-06", "2020-11-07", "2020-11-08", "2020-11-09", "2020-11-10", "2020-11-11", "2020-11-12", "2020-11-13", "2020-11-14", "2020-11-15", "2020-11-16", "2020-11-17", "2020-11-18", "2020-11-19", "2020-11-20", "2020-11-21", "2020-11-22", "2020-11-23", "2020-11-24", "2020-11-25", "2020-11-26", "2020-11-27", "2020-11-28", "2020-11-29", "2020-11-30", "2020-12-01", "2020-12-02", "2020-12-03", "2020-12-04", "2020-12-05", "2020-12-06", "2020-12-07", "2020-12-08", "2020-12-09", "2020-12-10", "2020-12-11", "2020-12-12", "2020-12-13", "2020-12-14", "2020-12-15", "2020-12-16", "2020-12-17", "2020-12-18", "2020-12-19",
                    "2020-12-20"
                 ]
    epi_dates = epi_dates[:config["num_dates"]]



    
    print("Loading data...")
    data_set = GraphRNN_dataset(epi_dates = epi_dates,
                                flow_dataset = flow_dataset,
                                epi_dataset = epi_dataset,
                                input_hor=config["input_hor"],
                                pred_hor=config["pred_hor"],
                                fake_data=False)
    visualize = False
    if visualize:
        data_set.visualize(0, num_nodes=5, num_edges=5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data_sampler = GraphRNN_DataSampler(data_set, input_hor=input_hor, pred_hor=pred_hor)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=config["batch_size"], pin_memory=True, num_workers=0, shuffle=True)
    
    print("Data loaded.")
    input_edge_weights, input_node_data, target_edge_weights, target_node_data = next(iter(data_loader))
    
    model  = Graph_RNN(n_nodes = data_set.n_nodes,
                       n_features = data_set.n_features,
                       h_size = config["h_size"],
                       f_out_size =config["h_size"],
                       fixed_edge_weights = input_edge_weights[0,0,:,:],
                       device=device,
                       dtype=torch.float32,
                       use_neighbors=config["use_neighbors"]
                       )
    
    

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    torch.autograd.set_detect_anomaly(False)

    model.to(device)
    profile = False
    if profile:
        log_dir = './log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Use torch.profiler to profile the model
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=False,
            with_stack=True
        ) as prof:
            print("Starting training with profiling...")
            losses, parameter_mag = train(model, data_loader,
                                    criterion, optimizer,
                                    config["pred_hor"], device, n_epochs=config["n_epochs"],
                                    save_name="model_state_dict.pth")
            print("Finished training with profiling.")
            # Verify that the log directory is populated
        if os.listdir(log_dir):
            print(f"Log files generated in {log_dir}")
        else:
            print(f"No log files found in {log_dir}")
    else:
        losses, parameter_mag, gradients, hidden_state_mag = train(model, data_loader,
                        criterion, optimizer,
                        config["pred_hor"], device, 
                        n_epochs=config["n_epochs"],
                        max_grad_norm=config["max_grad_norm"],
                        save_name="test")


    
    print("Plotting losses...")
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()
    
    print("Plotting parameter magnitudes...")
    plt.figure()
    for param_name, param_mag in parameter_mag.items():
        plt.plot(param_mag, label=param_name)
    plt.plot(hidden_state_mag, label="Hidden State")
    
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Magnitude")
    # plt.yscale("log")
    plt.legend()
    plt.show()
    
    plt.figure()
    for param_name, grad in gradients["pre_limit"].items():
        plt.plot(grad, label=f"pre {param_name}" , linestyle="--")
    for param_name, grad in gradients["post_limit"].items():
        plt.plot(grad, label=f"post {param_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Magnitude")
    plt.legend()
    plt.show()
    print("Finished training run.")
    
    
