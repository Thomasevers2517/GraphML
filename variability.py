import pandas as pd
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

dataset = []
small_dataset = []
nodes_subset_data = []
initial_node_data = {}
all_timestamp_node_data = []
subset_nodes = None

def load_data():
    for i in range(10):
        if i < 9:
            j = i + 1
            data = pd.read_csv('data/daily_county2county_2019_01_0' + str(j) + '.csv')
        else:
            j = i + 1
            data = pd.read_csv('data/daily_county2county_2019_01_' + str(j) + '.csv')
        dataset.append(data)
    return dataset

def create_small_dataset(dataset):
    nodes = dataset[0][['geoid_o', 'lat_o', 'lng_o']].drop_duplicates().reset_index(drop=True)
    subset_nodes = nodes[(nodes['lng_o'] > -90) & (nodes['lng_o'] < -70) & (nodes['lat_o'] > 25) & (nodes['lat_o'] < 30)]

    for i in range(10):
        data = dataset[i]
        subset_data = data[data['geoid_o'].isin(subset_nodes['geoid_o']) & data['geoid_d'].isin(subset_nodes['geoid_o'])]
        small_dataset.append(subset_data)
    return subset_nodes, small_dataset

def process_small_dataset(subset_nodes, small_dataset):
    for i in range(3):
        print("NEW SECTION")
        print(small_dataset[i])
        small_dataset[i] = small_dataset[i][small_dataset[i]['geoid_o'] != small_dataset[i]['geoid_d']]
        G = nx.from_pandas_edgelist(small_dataset[i], 'geoid_o', 'geoid_d', ['pop_flows'])
        pos = {row['geoid_o']: (row['lng_o'], row['lat_o']) for i, row in subset_nodes.iterrows()} 
        nx.draw(G, pos, edge_color=[G[u][v]['pop_flows'] for u, v in sorted(G.edges, key=lambda x: G[x[0]][x[1]]['pop_flows'], reverse=False)], edge_cmap=plt.cm.Blues, node_size=7)
        plt.show()

def initialize_node_data(subset_nodes, small_dataset):
    first_day_data = small_dataset[0]
    nodes_subset_data = small_dataset[0]['geoid_o'].unique()
    for node in nodes_subset_data:
        initial_node_data[node] = round(first_day_data[first_day_data['geoid_o'] == node]['pop_flows'].values[0] 
                                        / first_day_data[first_day_data['geoid_o'] == node]['visitor_flows'].values[0], 2)
    print(initial_node_data)
    return initial_node_data

def update_node_data(initial_node_data):
    current_node_data = initial_node_data.copy()
    for i in range(len(small_dataset)):
        data = small_dataset[i]
        for index, row in data.iterrows():
            current_node_data[row['geoid_o']] -= row['pop_flows']
            current_node_data[row['geoid_o']] = int(current_node_data[row['geoid_o']])
            current_node_data[row['geoid_d']] += row['pop_flows']
            current_node_data[row['geoid_d']] = int(current_node_data[row['geoid_d']])
        print("NEW DAY: ", i+1)
        print(current_node_data)
        all_timestamp_node_data.append(current_node_data)
    return all_timestamp_node_data

def plot_node_values(subset_nodes, all_timestamp_node_data):
    for i in range(len(all_timestamp_node_data)):
        plt.scatter(subset_nodes['lng_o'], subset_nodes['lat_o'], c=list(all_timestamp_node_data[i].values()), cmap='hot')
        plt.show()

def main():
    dataset = load_data()
    subset_nodes, small_dataset = create_small_dataset(dataset)
    
    # plot the small dataset to make sure operates as intended
    process_small_dataset(subset_nodes, small_dataset)

    # initialize node data in the visualization process, actually not needed in the end (only looking at edge information)
    initial_node_data = initialize_node_data(subset_nodes, small_dataset)

    # update node data for each timestamp , actually not needed in the end (only looking at edge information)
    all_timestamp_node_data = update_node_data(initial_node_data)

    # actually not needed in the end (only looking at edge information)
    plot_node_values(subset_nodes, all_timestamp_node_data)

if __name__ == "__main__":
    main()
