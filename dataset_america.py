import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def load_data(file_path):
    return pd.read_csv(file_path)

def normalize(data, column):
    weights = data[column]
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    weight_range = max_weight - min_weight
    normalized_edge_weights = (weights - min_weight) / weight_range
    data[column] = normalized_edge_weights
    return normalized_edge_weights

def skew_towards_maximum(data, column, skew=10):
    weights = data[column]
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    weight_range = max_weight - min_weight
    shifted = weights - min_weight + 1
    log_values = np.log(shifted) / np.log(skew)
    skewed_weights = (log_values / np.max(log_values)) * weight_range + min_weight
    data[column] = skewed_weights
    return skewed_weights

def preprocess_data(data):
    geoid_indexed_data = data[['geoid_o', 'lat_o', 'lng_o']].drop_duplicates(subset=['geoid_o']).set_index('geoid_o')
    geoid_d_indexed_data = data[['geoid_d', 'lat_d', 'lng_d']].drop_duplicates(subset=['geoid_d']).set_index('geoid_d').rename(columns={'lat_d': 'lat_o', 'lng_d': 'lng_o'})
    geoid_indexed_data.update(geoid_d_indexed_data)
    return geoid_indexed_data

def plot_nodes(ax, geoid_indexed_data):
    lngs = geoid_indexed_data['lng_o']
    lats = geoid_indexed_data['lat_o']
    ax.scatter(lngs, lats, color='red', s=0.1)

def plot_edges(ax, data, weights):
    starts = [data['lng_o'], data['lat_o']]
    ends = [data['lng_d'], data['lat_d']]
    lines = np.array(list(zip(starts, ends))).T
    lines = LineCollection(lines, linewidths=.1*weights)
    ax.add_collection(lines)

def filter_and_normalize_data(data):
    data = data.query('geoid_o != geoid_d')
    normalize(data, 'pop_flows')
    skew_towards_maximum(data, 'visitor_flows', skew=10)
    normalize(data, 'visitor_flows')
    data = data.sort_values('visitor_flows').groupby('geoid_o').head(1)
    return data

def build_graph(data):
    graph = nx.from_pandas_edgelist(data, 'geoid_o', 'geoid_d', ['visitor_flows'])
    return graph, nx.adjacency_matrix(graph)

def calculate_sparsity(graph):
    M = nx.to_scipy_sparse_array(graph)
    matrix_size = len(graph.nodes) ** 2
    sparsity = (matrix_size - M.nnz) / matrix_size
    return sparsity

def draw_network(coo_matrix, node_size=10, width=0.1):
    graph = nx.from_numpy_array(coo_matrix.toarray())
    nx.draw_networkx(graph, node_size=node_size, width=width, with_labels=False)
    print(f"Sparsity: {calculate_sparsity(graph)}")
    plt.show()

def create_time_graph(S, timegraph_size=2):
    St = np.zeros((timegraph_size, timegraph_size))
    for i in range(1, timegraph_size):
        St[i, i-1] = 1
    return St, sp.sparse.kron(St, S), sp.sparse.kron(St, np.identity(S.shape[0])) + sp.sparse.kron(np.identity(St.shape[0]), S)

def main():
    file_path = 'daily_county2county_2019_01_01.csv'
    data = load_data(file_path)
    geoid_indexed_data = preprocess_data(data)
    
    fig, ax = plt.subplots()
    plot_nodes(ax, geoid_indexed_data)
    
    data = filter_and_normalize_data(data)
    plot_edges(ax, data, data['visitor_flows'])
    
    plt.show()
    
    graph, S = build_graph(data)
    print(f"Graph is connected: {nx.is_connected(graph)}")
    
    timegraph_size = 2
    St, kronecker_product, cartesian_product = create_time_graph(S, timegraph_size)
    
    strong_product = kronecker_product + cartesian_product
    draw_network(strong_product)

if __name__ == "__main__":
    main()
