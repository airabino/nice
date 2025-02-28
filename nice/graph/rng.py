import numpy as np
import networkx as nx

from .graph import graph_from_nlg
from .utilities import pythagorean
from .floyd_warshall import shortest_paths

default_rng = np.random.default_rng()

def connected_nk_graph(n, k, **kwargs):
    '''
    Generate a connected n-k graph

    The resulting graph will 

    rng = kwargs.get('rng', default_rng)
    label = kwargs.get('label', '')

    x_gen = kwargs.get('x', lambda n: rng.uniform(0, 1, size = (n, )))
    y_gen = kwargs.get('y', lambda n: rng.uniform(0, 1, size = (n, )))

    node_functions = kwargs.get('node_functions', {})
    edge_functions = kwargs.get('edge_functions', {})

    x = x_gen(n)
    y = y_gen(n)

    x_s, x_t = np.meshgrid(x, x, indexing = 'ij')
    y_s, y_t = np.meshgrid(y, y, indexing = 'ij')

    distance = pythagorean(x_s, y_s, x_t, y_t)

    ccg = nx.from_numpy_array(distance, edge_attr = 'distance')

    mst = nx.minimum_spanning_tree(ccg, weight = 'distance')
    
    nodes = []

    for idx in range(n):

        node = {
            'id': f'{label}_{idx}',
            'x': x[idx],
            'y': y[idx],
            'n': idx,
        }

        for key, function in node_functions.items():

            node[key] = function(node)

        nodes.append(node)

    links = []
    node_link_counts = {n['id']: 0 for n in nodes}

    for idx_s in range(n):

        mse = list(mst._adj[nodes[idx_s]['n']].keys())

        indices = np.argsort(distance[nodes[idx_s]['n']])[1:]

        for idx_t in mse:

            source = nodes[idx_s]['id']
            target = nodes[idx_t]['id']

            d = distance[idx_s, idx_t]

            edge = {
                'source': source,
                'target': target,
                'distance': d,
            }

            for key, function in edge_functions.items():

                edge[key] = function(edge)

            links.append(edge)

            node_link_counts[source] += 1

        for idx_t in indices:

            if idx_t in mse:

                continue

            if node_link_counts[source] > k:

                break

            source = nodes[idx_s]['id']
            target = nodes[idx_t]['id']

            d = distance[idx_s, idx_t]

            edge = {
                'source': source,
                'target': target,
                'distance': d,
            }

            for key, function in edge_functions.items():

                edge[key] = function(edge)

            links.append(edge)

            node_link_counts[source] += 1

    return graph_from_nlg({'nodes': nodes, 'links': links}, **kwargs.get('nx', {}))

def all_pairs_graph(graph, **kwargs):

    conditions = kwargs.get('conditions', [])

    predecessors, values, paths = shortest_paths(graph, **kwargs)

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