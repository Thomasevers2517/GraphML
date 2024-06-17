
# for a single instant in time, build a graph with nodes as counties and edges as mobility between counties

import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from functools import reduce
import os
import sys

class Preprocessor:
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
    def __init__(self, flow_dataset, epi_dataset, epi_dates, k=3, skew=10, plottable=False):
        self.timegraph_size = len(epi_dates)
        self.k = k
        self.skew = skew
        self.plottable = plottable

        # load in data
        self.epidemiology = pd.read_csv(epi_dataset, keep_default_na=False, na_values=[""])
        self.flow = pd.read_csv(flow_dataset)

        # preprocess
        self.process_epidemiology()
        self.epidemiology_timesteps = [self.epi_extract_date(date) for date in epi_dates]
        self.process_population_flow()

        # Ensure nodes exist in all time steps
        all_epidemiology_geoids = reduce(np.intersect1d, [timestep['geoid_o'].unique() for timestep in self.epidemiology_timesteps])
        all_flow_geoids = np.union1d(self.flow['geoid_o'].unique(), self.flow['geoid_d'].unique())
        intersection = np.intersect1d(all_epidemiology_geoids, all_flow_geoids)
        
        for i, epi in enumerate(self.epidemiology_timesteps):
            self.epidemiology_timesteps[i] = epi[epi['geoid_o'].isin(intersection)]
        
        self.flow = self.flow[self.flow['geoid_o'].isin(intersection) & self.flow['geoid_d'].isin(intersection)]



    def normalize(self, column):
        weights = self.flow[column]
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        weight_range = max_weight - min_weight
        normalized_edge_weights = (weights - min_weight) / weight_range

        # Set the values directly
        self.flow[column] = normalized_edge_weights

    def skew_towards_maximum(self, column):
        weights = self.flow[column]
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        weight_range = max_weight - min_weight

        # Perform the skew
        shifted = weights - min_weight + 1  # Shift values to avoid log(0)
        log_values = np.log(shifted) / np.log(self.skew)

        # Scale the log-transformed values back to the original range
        skewed_weights = (log_values / np.max(log_values)) * weight_range + min_weight

        # Set the values directly
        self.flow[column] = skewed_weights

    def process_epidemiology(self):
        self.epidemiology = self.epidemiology[self.epidemiology['location_key'].str.match("^US_.*_\d+")]
        self.epidemiology["geoid_o"] = self.epidemiology['location_key'].str.extract(r'US_.*_(\d+)').astype('int64')
        self.epidemiology['infection_gone'] = self.epidemiology[['new_deceased','new_recovered']].sum(axis=1)
        self.epidemiology = self.epidemiology[['geoid_o', 'date', 'new_confirmed', 'infection_gone']]

    def epi_extract_date(self, date):
        data = self.epidemiology.copy()
        data = data[data['date'] == date]
        return data

    def process_population_flow(self):
        # index by geoid of both target and origin, and merge them to we ensure we have a list of all nodes
        geoid_o_indexed_data = self.flow[['geoid_o', 'lat_o', 'lng_o']].drop_duplicates(subset=['geoid_o']).set_index('geoid_o')
        geoid_d_indexed_data = self.flow[['geoid_d', 'lat_d', 'lng_d']].drop_duplicates(subset=['geoid_d']).set_index('geoid_d').rename(index={'geoid_d': 'geoid_o'}).rename(index={'lat_d': 'lat_o'}).rename(index={'lng_d': 'lng_o'})

        geoid_indexed_data = geoid_o_indexed_data.combine_first(geoid_d_indexed_data)

        # filter out selfloops
        self.flow = self.flow.query('geoid_o != geoid_d')

        #normalize population flows
        self.skew_towards_maximum('pop_flows')
        self.normalize('pop_flows')

        #normalize visitor flows
        self.skew_towards_maximum('visitor_flows')
        self.normalize('visitor_flows')

        # only show data for visitor flows above some threshold
        #data = data.query('visitor_flows > 0.15')

        # keep k highest value edges for each from_node and each to_node
        data_from = self.flow.sort_values('visitor_flows').groupby('geoid_o').tail(self.k)
        data_to = self.flow.sort_values('visitor_flows').groupby('geoid_d').tail(self.k)
        self.flow = data_from.combine_first(data_to)

        #TODO:  keep edges such that the top 80% of population_flow per node is accounted for

    def kronecker(self):
        print("use manual_kronecker instead")
        exit(0)

        # Get adjacency matrix for out graph
        graph = nx.from_pandas_edgelist(self.flow, 'geoid_o', 'geoid_d', ['visitor_flows'])
        S = nx.adjacency_matrix(graph)

        # create timegraph
        St = np.zeros((self.timegraph_size,self.timegraph_size))
        for i in range(1, self.timegraph_size):
            St[i, i-1] = 1

        # Kronecker
        kronecker_product = sp.sparse.kron(St, S)

        # make a dataframe out of it
        kron_graph = nx.from_numpy_array(kron.toarray())
        kron_df = nx.to_pandas_edgelist(kron_graph, 'from', 'to')

        # Match epidemiology nodes with population flow nodes
        node_signal_flows = [self.flow.merge(signal, on="geoid_o") for signal in self.epidemiology_timesteps]

        # hstack epidemiological data so it has the same shape as the kronecker graph
        node_signal_flows = pd.concat(node_signals_flows, ignore_index=True)

        # vstack kronecker graph with the epidemiological data
        self.flow = pd.concat([kron_df, node_signal_flows], axis=1)

    def combined_manual_kronecker(self):
        geoid_offset = 1000000
        timesteps = []
        for i in range(self.timegraph_size - 1):
            timestep_data = self.flow.copy()\
                .merge(self.epidemiology_timesteps[i], on="geoid_o")
                #.merge(self.epidemiology_timesteps[i+1].copy().rename(index={'geoid_o': 'geoid_d'}), on="geoid_d") # if we want the destination nodes to also have signals
            timestep_data['timestep'] = i
            timestep_data['geoid_o'] += geoid_offset * i
            timestep_data['geoid_d'] += geoid_offset * (i+1)
            if (self.plottable):
                timestep_data['lng_o'] += 150 * i
                timestep_data['lng_d'] += 150 * (i+1)
            timesteps.append(timestep_data)

        return pd.concat(timesteps, ignore_index=True)

    def disjoint_manual_kronecker(self):
        geoid_offset = 1000000
        timesteps = []
        for i in range(self.timegraph_size):
            # update epi geoid
            self.epidemiology_timesteps[i]['geoid_o'] += geoid_offset * i

        for i in range(self.timegraph_size - 1):
            # Updae flow geoid
            timestep_data = self.flow.copy()
            timestep_data['timestep'] = i
            timestep_data['geoid_o'] += geoid_offset * i
            timestep_data['geoid_d'] += geoid_offset * (i+1)
            timesteps.append(timestep_data)

        return pd.concat(timesteps, ignore_index=True), pd.concat(self.epidemiology_timesteps)

    def get_data_for_graphRNN(self): 
        flow = self.flow
        epidemiology_timesteps = self.epidemiology_timesteps
        return flow, epidemiology_timesteps
    
def draw_network(data, weight_name='visitor_flows', node_size=1, line_width_mod=0.1):
    # initialize plot
    fig, ax = plt.subplots()

    # plot the from_nodes
    lngs = data['lng_o']
    lats = data['lat_o']
    ax.scatter(lngs, lats, color='red', s=node_size)

    # plot the to_nodes
    lngs = data['lng_d']
    lats = data['lat_d']
    ax.scatter(lngs, lats, color='red', s=node_size)

    # plot the edges
    starts = [data['lng_o'], data['lat_o']]
    ends = [data['lng_d'], data['lat_d']]
    weights = data[weight_name]
    lines = np.array(list(zip(starts, ends))).T
    lines = LineCollection(lines, linewidths=line_width_mod*weights)
    ax.add_collection(lines)

    ax.autoscale()
    plt.show()

if __name__ == "__main__":

    flow_dataset = "data/daily_county2county_2019_01_01.csv"
    epi_dataset = "data_epi/epidemiology.csv"
    epi_dates = ["2020-06-09", "2020-06-10"]
    preprocessor = Preprocessor(flow_dataset, epi_dataset, epi_dates, plottable=True)

    graph_df = preprocessor.combined_manual_kronecker()
    kron_flow_df, signals_df = preprocessor.disjoint_manual_kronecker()

    print(kron_flow_df.shape)
    print(signals_df.shape)

    draw_network(graph_df)
