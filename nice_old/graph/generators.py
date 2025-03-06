import numpy as np
import networkx as nx

from scipy.stats import poisson
from heapq import heappop, heappush
from itertools import count

# from networkx.generators import *

from ..utilities import pythagorean
from ..routing import all_pairs_shortest_paths

default_rng = np.random.default_rng()

def nmu_graph(n, mu, **kwargs):
    '''
    Generate a connected n-k graph

    The graph is defined by a set of n randomly generated points. These points are connected
    by a minimum spanning tree. Following this, the points are connected to the k nearest
    neighbors such that all points have k connections
    '''

    rng = kwargs.get('rng', default_rng)
    xlim = kwargs.get('xlim', (0, 1))
    ylim = kwargs.get('xlim', (0, 1))
    x = kwargs.get('x', rng.uniform(*xlim, size = (n, )))
    y = kwargs.get('y', rng.uniform(*ylim, size = (n, )))
    _class = kwargs.get('create_using', nx.Graph)

    x_s, x_t = np.meshgrid(x, x, indexing = 'ij')
    y_s, y_t = np.meshgrid(y, y, indexing = 'ij')

    distance = pythagorean(x_s, y_s, x_t, y_t)

    ccg = nx.from_numpy_array(distance, edge_attr = 'distance')

    mst = nx.minimum_spanning_tree(ccg, weight = 'distance')
    
    nodes = []

    for idx in range(n):

        handle = idx

        node = {
            'x': x[idx],
            'y': y[idx],
        }

        nodes.append((handle, node))

    edges = []

    degree = {idx: 0 for idx in range(n)}

    for source, _adj in mst._adj.items():

        distance[source][source] = np.inf

        for target, edge in _adj.items():

            degree[source] += 1
            distance[source][target] = np.inf

            edges.append((source, target, edge))

    heap = []
    c = count()

    k_dist = poisson(mu)

    for source, deg in degree.items():

        heappush(heap, (k_dist.rvs(random_state = rng) - deg, next(c), source))

    while heap:

        degree, _, source = heappop(heap)

        if degree <= 0:

            continue

        target = np.argsort(distance[source])[0]

        edges.append((source, target, {'distance': distance[source][target]}))

    graph = _class()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph

def nlh_graph(n, l = 0, h = np.inf, **kwargs):

    rng = kwargs.get('rng', default_rng)
    xlim = kwargs.get('xlim', (0, 1))
    ylim = kwargs.get('xlim', (0, 1))
    x = kwargs.get('x', rng.uniform(*xlim, size = (n, )))
    y = kwargs.get('y', rng.uniform(*ylim, size = (n, )))
    _class = kwargs.get('create_using', nx.Graph)

    x_s, x_t = np.meshgrid(x, x, indexing = 'ij')
    y_s, y_t = np.meshgrid(y, y, indexing = 'ij')

    distance = pythagorean(x_s, y_s, x_t, y_t)

    nodes = []

    for idx in range(n):

        handle = idx

        node = {
            'x': x[idx],
            'y': y[idx],
        }

        nodes.append((handle, node))

    edges = []

    for source in range(n):
        for target in range(n):

            d = distance[source][target]

            if (d >= l) and (d <= h):

                edges.append((source, target, {'distance': d}))

    graph = _class()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph

def all_pairs_graph(graph, conditions = [], simple = False, **kwargs):

    if simple:

        x = [n['x'] for n in graph._node.values()]
        y = [n['y'] for n in graph._node.values()]
        keys = list(graph.nodes())

        x_s, x_t = np.meshgrid(x, x, indexing = 'ij')
        y_s, y_t = np.meshgrid(y, y, indexing = 'ij')

        distances = pythagorean(x_s, y_s, x_t, y_t)

        values = {}

        for i, s in enumerate(keys):

            values[s] = {}

            for j, t in enumerate(keys):

                values[s][t] = {'distance': distances[i, j]}

    else:

        _, values, _ = all_pairs_shortest_paths(graph, **kwargs)

    nodes = [(k, v) for k, v in graph._node.items()]

    edges = []

    for source, adj in values.items():
        for target, val in adj.items():
            # if source != target:

            feasible = np.product([fun(val) for fun in conditions])

            if feasible:

                edges.append((source, target, val))

    apg = nx.DiGraph()
    apg.add_nodes_from(nodes)
    apg.add_edges_from(edges)

    return apg