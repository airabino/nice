import json
import momepy
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

from heapq import heappush, heappop
from itertools import count, islice
from networkx.algorithms.simple_paths import _all_simple_edge_paths

from ..progress_bar import ProgressBar

def get_paths(graph, **kwargs):

    terminals = kwargs.get('terminals', None)
    n_paths = kwargs.get('n_paths', None)
    floor = kwargs.get('floor', None)
    objective = kwargs.get('objective', None)
    weight = kwargs.get('weight', None)

    paths = []

    if terminals is None:

        terminals = list(graph.nodes())

    for origin in ProgressBar(terminals):

        destinations = set(terminals) - set([origin])

        lg = level_graph(graph, origin, objective = objective)

        for destination in list(destinations):

            if lg._node[destination]['cost'] < floor:

                continue

            if np.isinf(lg._node[destination]['cost']):
                
                continue

            ksp = k_shortest_paths(
                lg, origin, destination, k = n_paths, objective = weight
                )

            for idx, path in enumerate(ksp):

                paths.append(
                    {
                        'origin': origin,
                        'destination': destination,
                        'index': idx,
                        'path': path,
                        }
                    )

    return paths

def get_raw_paths(graph, origins, **kwargs):

    cutoff = kwargs.get('cutoff', 4)
    objective = kwargs.get('objective', None)
    min_edge_cost = kwargs.get('min_edge_cost', 0)

    raw_paths = []

    for origin in ProgressBar(origins):

        destinations = list(set(origins) - set([origin]))

        lg = level_graph(
            graph,
            origin,
            objective = objective,
            min_edge_cost = min_edge_cost,
            terminals = destinations,
        )

        asep = list(
            _all_simple_edge_paths(lg, origin, set(destinations), cutoff = cutoff)
            )

        raw_paths.extend(asep)

    return raw_paths




def level_graph(graph, origin, **kwargs):

    terminals = kwargs.get('terminals', [])
    objective = kwargs.get('objective', None)
    min_edge_cost = kwargs.get('min_edge_cost', 0)

    _node = graph._node
    _adj = graph._adj

    costs = {k: np.inf for k in graph.nodes}
    visited = []

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    heappush(heap, (0, next(c), origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost, k, source = heappop(heap)

        if source in visited:

            continue  # already searched this node.

        visited.append(source)

        costs[source] = cost

        for target, edge in _adj[source].items():

            # Updating states for edge traversal
            cost_edge = edge.get(objective, 1)

            cost_target = cost + cost_edge

            # Updating the weighted cost for the path
            savings = cost_target < costs.get(target, np.inf)

            if savings:

                heappush(heap, (cost_target, next(c), target))


    nodes = []
    edges = []

    for source, node in _node.items():

        node['cost'] = costs[source]

        nodes.append((source, node))

        if source in terminals:

            continue
        
        for target, edge in graph._adj[source].items():

            if costs[target] > costs[source] + min_edge_cost:

                edges.append((source, target, edge))

    level_graph = nx.DiGraph()
    level_graph.add_nodes_from(nodes)
    level_graph.add_edges_from(edges)

    return level_graph

def k_shortest_paths(graph, origin, destination, k = None, objective = None):

    path_gen = nx.shortest_simple_paths(
        graph, origin, destination, weight = objective
        )

    paths = list(islice(path_gen, k))

    return paths