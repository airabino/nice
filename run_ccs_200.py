import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt
import pyomo.environ as pe

from scipy.stats import rv_histogram

import nice

'''
Loading
'''
# graph_all = nice.graph.graph_from_json('Data/graph_100k.json')
# graph_ccs = nice.graph.graph_from_json('Data/graph_ccs_100k.json')
graph = nice.graph.graph_from_json('Data/graph_ccs_200.json')

# paths_all = json.load(open('Data/paths_100k.json', 'r'))
# paths_ccs = json.load(open('Data/paths_ccs_100k.json', 'r'))
paths = json.load(open('Data/paths_ccs_100k.json', 'r'))

# save_path = '/media/aaron/Extreme SSD/nice_results/all/'
# save_path = '/media/aaron/Extreme SSD/nice_results/ccs/'
save_path = '/media/aaron/Extreme SSD/nice_results/ccs_200/'


for source, _adj in graph._adj.items():
    for target, edge in _adj.items():

        edge['cost'] = edge['time']

places = [k for k, n in graph._node.items() if 'station' not in k]
stations = [k for k, n in graph._node.items() if 'station' in k]
paths = [p for p in paths if len(p['path']) > 2]

threshold = 20 * 60

local_stations = []

for source in places:
    for target in stations:
        if graph._adj[source].get(target, {'cost': np.inf})['cost'] <= threshold:

            local_stations.append(target)

'''
Setting travel demand
'''

p = [ 0.00890009, -0.00686443]
f = lambda d: p[0] * np.exp(p[1] * d)

kw = {
    'routing_weight': 'time',
    'production': 'population',
}

graph = nice.demand.demand(graph, places, **kw)

'''
Adding charging information at stations
'''

energy = 35 * 3.6e6
power = 80e3
m = 1 / (energy / power)
rho = np.concatenate((np.linspace(0, .8, 2), np.linspace(.81, .99, 20)))

for station in stations:

    node = graph._node[station]

    power = node['power_kw'] * 1e3
    m = 1 / (energy / power)
    c = [node['port_count']]
    queue = nice.queue.Queue(m = m, rho = rho, c = c)
    
    volumes = np.array([rho * m * size for size in c])
    volume1 = rho * m * c1
    delays =  np.array(
        [queue.interpolate(rho, size)[0] * rho * m * size for size in c]
    )

    base_volume = (
        .5 * volume1.max() * (station in local_stations)
    )

    graph._node[station]['power'] = power
    graph._node[station]['volumes'] = np.atleast_2d(volumes)
    graph._node[station]['delays'] = np.atleast_2d(delays * volumes)
    graph._node[station]['counts'] = c
    graph._node[station]['expenditures'] = [0]
    graph._node[station]['base_volume'] = base_volume

'''
Adding classes and Building the network
'''

for place in places:

    graph._node[place]['_class'] = nice.optimization.Place

for station in stations:

    graph._node[station]['_class'] = nice.optimization.Station

for source, _adj in graph._adj.items():
    for target, edge in _adj.items():

        edge['cost'] = edge['time']
        edge['_class'] = nice.optimization.Edge

for path in paths:
    
    path['_class'] = nice.optimization.Path

kw = {
    'verbose': True,
}

network = nice.optimization.Network(**kw).from_graph(graph, paths)

network.build()

'''
Solving the model
'''

scales = np.arange(1e3, 1e5 + 1e3, 1e3) / 3600
costs = []
ratios = []

k = 0

for scale in nice.progress_bar.ProgressBar(scales):

    kw = {
        'verbose': False,
        'solver': {
            '_name': 'appsi_highs',
            'time_limit': 10 * 6,
        },
    }
    
    network.model.scale = scale
    network.solve(**kw)
    solution = network.solution

    num = sum(
        [n.get('mode_switch', 0) for n in solution._node.values()]
    )
    
    den = sum(
        [n.get('total', 0) for n in solution._node.values()]
    )
    
    ratio = np.nan_to_num(num / den)

    ratios.append(ratio)
    costs.append(network.objective_value)

    nice.graph.graph_to_json(solution, save_path + f"run_{k}_4.json")

    k += 1

nice.utilities.to_json(
    {'scales': scales, 'costs': costs, 'ratios': ratios},
    save_path + "summary_4.json",
    )