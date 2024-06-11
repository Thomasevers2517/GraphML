
# for a single instant in time, build a graph with nodes as counties and edges as mobility between counties

import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import sys

# load data
data = pd.read_csv('daily_county2county_2019_01_01.csv')

''' 
Daily Flow Data 
There are 594160 links of which across each link the following is noted
geoid_o - Unique identifier of the origin geographic unit (census tract, county, and state). Type: string.
geoid_d - Unique identifier of the destination geographic unit (census tract, county, and state). Type: string.
lat_o - Latitude of the geometric centroid of the origin unit. Type: float.
lng_o - Longitude of the geometric centroid of the origin unit. Type: float.
lat_d - Latitude of the geometric centroid of the destination unit. Type: float.
lng_d - Longitude of the geometric centroid of the destination unit. Type: float.
date - Date of the records. Type: string.
visitor_flows - Estimated number of visitors between the two geographic units (from geoid_o to geoid_d). Type: float.
pop_flows - Estimated population flows between the two geographic units (from geoid_o to geoid_d), inferred from visitor_flows. Type: float.
'''
def normalize(column):
    weights = data[column]
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    weight_range = max_weight - min_weight
    normalized_edge_weights = (weights - min_weight) / weight_range

    # Set the values directly and return them
    data[column] = normalized_edge_weights
    return normalized_edge_weights

def skew_towards_maximum(column, skew=10):
    weights = data[column]
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    weight_range = max_weight - min_weight

    # Perform the skew
    shifted = weights - min_weight + 1  # Shift values to avoid log(0)
    log_values = np.log(shifted) / np.log(skew)

    # Scale the log-transformed values back to the original range
    skewed_weights = (log_values / np.max(log_values)) * weight_range + min_weight

    # Set the values directly and return them
    data[column] = skewed_weights
    return skewed_weights

# initialize plot
fig, ax = plt.subplots()

# index by geoid of both target and origin, and merge them to we ensure we have a list of all nodes
geoid_indexed_data = data[['geoid_o', 'lat_o', 'lng_o']].drop_duplicates(subset=['geoid_o']).set_index('geoid_o')
geoid_d_indexed_data = data[['geoid_d', 'lat_d', 'lng_d']].drop_duplicates(subset=['geoid_d']).set_index('geoid_d').rename(index={'geoid_d': 'geoid_o'}).rename(index={'lat_d': 'lat_o'}).rename(index={'lng_d': 'lng_o'})

geoid_indexed_data.update(geoid_d_indexed_data)

# plot the nodes
lngs = geoid_indexed_data['lng_o']
lats = geoid_indexed_data['lat_o']
ax.scatter(lngs, lats, color='red', s=0.1)

# filter out selfloops
data = data.query('geoid_o != geoid_d')

#normalize population flows
normalize('pop_flows')

#normalize visitor flows
weights = skew_towards_maximum('visitor_flows', skew=10)
print(weights[:10])
weights = normalize('visitor_flows')
print(weights[:10])

# only show data for visitor flows above some threshold
#data = data.query('visitor_flows > 0.15')

# keep k highest value edges for each from_node
k=1
data = data.sort_values('visitor_flows').groupby('geoid_o').head(k)

#TODO:  keep edges such that the top 80% of population_flow per node is accounted for

# plot the edges
starts = [data['lng_o'], data['lat_o']]
ends = [data['lng_d'], data['lat_d']]
lines = np.array(list(zip(starts, ends))).T
lines = LineCollection(lines, linewidths=.1*weights)
ax.add_collection(lines)
plt.show()

# Get adjacency matrix for out graph
#graph.remove_edges_from(nx.selfloop_edges(G_whole))
graph = nx.from_pandas_edgelist(data, 'geoid_o', 'geoid_d', ['visitor_flows'])
S = nx.adjacency_matrix(graph)

# check if S is connected
print(nx.is_connected(graph))

def calculate_sparsity(graph):
  M = nx.to_scipy_sparse_array(graph)
  matrix_size = len(graph.nodes) ** 2
  sparsity = (matrix_size - M.nnz) / matrix_size
  return sparsity

def draw_network(coo_matrix, node_size=10, width=0.1):
    graph = nx.from_numpy_array(coo_matrix.toarray())
    nx.draw_networkx(graph, \
            node_size=node_size, width=width, with_labels=False)
    print(f"Sparsity: {calculate_sparsity(graph)}")
    plt.show()

# create timegraph
timegraph_size=2
St = np.zeros((timegraph_size,timegraph_size))
for i in range(1, timegraph_size):
    St[i, i-1] = 1

# Kronecker
kronecker_product = sp.sparse.kron(St, S)
# draw_network(kronecker_product)

# Cartesian
# From paper On Cartesian product of matrices by Deepak Sarma: Cartesian product of two square matrices Aand Bas A&B=A⊗J+J⊗B, where J is the all one matrix of appropriate order and ⊗is the Kronecker product
cartesian_product = sp.sparse.kron(St, np.identity(S.shape[0])) + sp.sparse.kron(np.identity(St.shape[0]), S)
# draw_network(cartesian_product)

# Strong
strong_product = kronecker_product + cartesian_product
# draw_network(strong_product)


