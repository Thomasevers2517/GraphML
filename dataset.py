# OBJECTIVE: To explore the mobility dataset

# for a single instant in time, build a graph with nodes as counties and edges as mobility between counties

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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

# inspect the data in the first 5 rows
print(data.head())
print(data)

# data preprocessing
# how many rows have missing values in them?
print("rows with missing values:", data.isnull().sum())

# get a list of nodes and their corresponding latitude and longitudes
nodes = data[['geoid_o', 'lat_o', 'lng_o']].drop_duplicates().reset_index(drop=True)
print(nodes)

# is there a node called 15005?
print("Node 15005 exists:", '15005' in nodes['geoid_o'].values)
print("Node 15005 exists:", '15005' in data['geoid_d'].values)

# visualize these nodes based on their geographic location
plt.scatter(nodes['lng_o'], nodes['lat_o'])
plt.title('Counties in the US')
plt.show()

# display a small geographical subset of the graph
# show the plot only over a certain range of lng_o and lat_o
subset_nodes = nodes[(nodes['lng_o'] > -90) & (nodes['lng_o'] < -70) & (nodes['lat_o'] > 25) & (nodes['lat_o'] < 30)]
# how many nodes in subset_nodes
print(subset_nodes.shape)
plt.title('Counties in Florida')
plt.scatter(subset_nodes['lng_o'], subset_nodes['lat_o'])
plt.show()

# create a graph using data
subset_data = data[data['geoid_o'].isin(subset_nodes['geoid_o']) & data['geoid_d'].isin(subset_nodes['geoid_o'])]
# create the graph using subset_data but use their lat_o and lat_d to determine node positions
G = nx.from_pandas_edgelist(subset_data, 'geoid_o', 'geoid_d', ['visitor_flows'])
# the positions of the nodes are given by the lat_o and lng_o columns
pos = {row['geoid_o']: (row['lng_o'], row['lat_o']) for i, row in subset_nodes.iterrows()} 

print("Number of nodes (counties) in subset:", G.number_of_nodes())
print("Number of edges (mobilities):", G.number_of_edges())
print("Number of connected components:", nx.number_connected_components(G))

# plot the graph with the nodes at their geographic positions
# visualize the graph where edge weights are a gradient based on their value
# nx.draw(G, pos, edge_color=[G[u][v]['visitor_flows'] for u, v in G.edges], edge_cmap=plt.cm.Blues, node_size=5)

# plot the distribution of visitor_flows values
#plt.hist([G[u][v]['visitor_flows'] for u, v in G.edges], bins=100)
#plt.title('Distribution of visitor_flows values')
#plt.show()

# this is including self loops -- basically those self loops dominate the distribution and then result in the graph not presenting any links
# iters = 0
# for u, v in G.edges:
#     if u in pos and v in pos:
#         if G[u][v]['visitor_flows']/max(G[u][v]['visitor_flows'] for u, v in G.edges) >= 0.1:
 #            iters += 1
 #            print("u:", u, "v:", v, "visitor_flows:", G[u][v]['visitor_flows']/max(G[u][v]['visitor_flows'] for u, v in G.edges))
 #            # plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='blue', alpha=G[u][v]['visitor_flows']/max(G[u][v]['visitor_flows'] for u, v in G.edges), linewidth=2)
 #            plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='blue', linewidth=2)

# plt.title('Mobility between counties in Florida (self loops incl)')
# print("Number of edges with visitor_flows >= 0.1:", iters)
# plt.show()


# remove self loops in this graph
G.remove_edges_from(nx.selfloop_edges(G))
#plt.hist([G[u][v]['visitor_flows'] for u, v in G.edges], bins=100)
#plt.title('Distribution of visitor_flows values after self loops removed')
#plt.show()

# QUESTION: Self loops? What would that mean in context of mobility? Mobility from a county to itself? It seems from the example that self loops are not graphed

iters = 0
# for all nodes in G, plot them with plt.scatter

for u in G.nodes:
    if u in pos:
        plt.scatter(pos[u][0], pos[u][1], color='red', s=2)

for u, v in G.edges:
    if u in pos and v in pos:
        if G[u][v]['visitor_flows']/max(G[u][v]['visitor_flows'] for u, v in G.edges) >= 0.05:
            iters += 1
            print("u:", u, "v:", v, "visitor_flows:", G[u][v]['visitor_flows']/max(G[u][v]['visitor_flows'] for u, v in G.edges))
            # plt.scatter(pos[u][0], pos[u][1], color='red', s=2)
            # plt.scatter(pos[v][0], pos[v][1], color='red', s=2)
            plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='blue', alpha=G[u][v]['visitor_flows']/max(G[u][v]['visitor_flows'] for u, v in G.edges), linewidth=2)
            # plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='blue', linewidth=2)

plt.title('Mobility between counties in Florida (remv self loops)')
print("Number of edges with visitor_flows >= 0.05:", iters)
plt.show()


# repeat the same for the entire US dataset
G_whole = nx.from_pandas_edgelist(data, 'geoid_o', 'geoid_d', ['visitor_flows'])
pos_whole = {row['geoid_o']: (row['lng_o'], row['lat_o']) for i, row in nodes.iterrows()}

# remove self loops in this graph
G_whole.remove_edges_from(nx.selfloop_edges(G_whole))

print("Number of nodes (counties) in whole dataset:", G_whole.number_of_nodes())
print("Number of edges (mobilities) in whole dataset:", G_whole.number_of_edges())
print("Number of connected components in whole dataset:", nx.number_connected_components(G_whole))

# plot the graph with the nodes at their geographic positions
# visualize the graph where edge weights are a gradient based on their value

max_visitor_flows = max(G_whole[u][v]['visitor_flows'] for u, v in G_whole.edges)

for u in G_whole.nodes:
    if u in pos_whole:
        plt.scatter(pos_whole[u][0], pos_whole[u][1], color='red', s=1)


iters = 0
for u, v in G_whole.edges:
    if u in pos_whole and v in pos_whole:
        if G_whole[u][v]['visitor_flows']/max_visitor_flows >= 0.01:
            iters += 1
            # print("u:", u, "v:", v, "visitor_flows:", G_whole[u][v]['visitor_flows']/max_visitor_flows)
            weight = (G_whole[u][v]['visitor_flows'])/(max_visitor_flows)
            if (weight > 0.3):
                weight = 0.99
            else:
                weight = 2 * weight
                if (weight > 1.0):
                    print("ERROR")
            # plt.scatter(pos_whole[u][0], pos_whole[u][1], color='red', s=2)
            # plt.scatter(pos_whole[v][0], pos_whole[v][1], color='red', s=2)
            plt.plot([pos_whole[u][0], pos_whole[v][0]], [pos_whole[u][1], pos_whole[v][1]], color='blue', alpha=weight, linewidth=2)
            if iters % 10 == 0:
                print("iters:", iters)


plt.title('Mobility between counties in the US')
plt.show()
