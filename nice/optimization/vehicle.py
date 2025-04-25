import sys
import time
# import pao

import numpy as np
import networkx as nx

from ..routing import all_pairs_shortest_paths

def transformed_graph(graph, nodes = None, conditions = [], **kwargs):

    if nodes is None:

        nodes = list(graph.nodes())

    _, values, paths = all_pairs_shortest_paths(graph, **kwargs)

    nodes_tg = [(k, graph._node[k]) for k in nodes]

    edges_tg = []

    for source in nodes:
        for target in nodes:

            val = values[source][target]
            # if source != target:

            feasible = np.product([fun(val) for fun in conditions])

            val['path'] = paths[source][target]

            if feasible:

                edges_tg.append((source, target, val))

    apg = nx.DiGraph()
    apg.add_nodes_from(nodes_tg)
    apg.add_edges_from(edges_tg)

    return apg

class Vehicle():

    def __init__(self, **kwargs):

        self.capacity = kwargs.get('capacity', 80 * 3.6e6)
        self.consumption = kwargs.get('consumption', 500)
        self.soc_min = kwargs.get('soc_min', .1)
        self.soc_max = kwargs.get('soc_max', 1.)

        self.usable_capacity = self.capacity * (self.soc_max - self.soc_min)

        self.fields = kwargs.get('fields', ['time', 'distance', 'energy'])

    def energy(self, graph, field = 'distance'):

        for source, _adj in graph._adj.items():
            for target, edge in _adj.items():

                edge['energy'] = edge[field] * self.consumption

        return graph

    def trim(self, graph, nodes = None):

        if nodes is None:

            nodes = list(graph.nodes)

        remove_nodes = list(set(list(graph.nodes)) - set(nodes))

        remove_edges = []

        for source, _adj in graph._adj.items():
            for target, edge in _adj.items():
                if edge['energy'] > self.soc_max * self.capacity:

                    remove_edges.append((source, target))

                elif edge['energy'] < self.soc_min * self.capacity:

                    remove_edges.append((source, target))

        graph.remove_edges_from(remove_edges)
        graph.remove_nodes_from(remove_nodes)

        return graph

    def transform(self, atlas, nodes = None):

        conditions = [
            lambda e: e['energy'] >= self.soc_min * self.capacity,
            lambda e: e['energy'] <= self.soc_max * self.capacity,
        ]

        return transformed_graph(atlas, nodes, conditions, fields = self.fields)