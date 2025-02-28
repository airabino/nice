import numpy as np
import networkx as nx

from numba import jit

def all_pairs_shortest_paths(graph, **kwargs):

    # Processing kwargs
    fields = kwargs.get('fields', ['objective'])
    
    kwargs['objective'] = fields[0]

    costs, predecessors = floyd_warshall(graph, **kwargs)

    paths = recover_paths(graph, predecessors, **kwargs)
    values = recover_values(graph, paths, **kwargs)

    return predecessors, values, paths

def floyd_warshall(graph, **kwargs):
    '''
    Implements the Floyd-Warshall algorithm for all-pairs routing

    args:

    graph is a NetworkX Graph
    fields is a list of edge attributes - the first one listed will be used for routing

    kwargs:

    origins - list of nodes in graph from which routes will start
    destinations - list of nodes in graph at which reoutes will end
    pivots - list of nodes in graph which can serve as intermediaries in routes

    if origins, destinations, or pivots are not provided then all nodes will be used

    tolerance - float threshold of disambiguation for selecting alterante paths

    if a non-zero tolerance is provided then alternate paths may be produced
    '''

    # Processing kwargs
    objective = kwargs.get('objective', 'objective')
    maximum_edge_cost = kwargs.get('maximum_edge_cost', np.inf)
    maximum_path_cost = kwargs.get('maximum_path_cost', np.inf)
    # field = kwargs.get('field', None)
    pivots = kwargs.get('pivots', list(graph.nodes))
    perturbation = kwargs.get('perturbation', 0)
    adjacency = kwargs.get('adjacency', None)

    if adjacency is None:

        # Creating adjacency matrix
        adjacency = nx.to_numpy_array(graph, weight = objective, nonedge = np.inf)
    
    adjacency[adjacency > maximum_edge_cost] = np.inf

    node_to_idx = {k: idx for idx, k in enumerate(graph.nodes)}
    idx_to_node = {idx: k for idx, k in enumerate(graph.nodes)}

    n = len(adjacency)

    if perturbation > 0:

        adjacency *= np.random.uniform(
            1 - perturbation,
            1 + perturbation,
            size = adjacency.shape
            )

    pivots_idx = [node_to_idx[k] for k in pivots]

    # Running the Floyd Warshall algorithm
    costs = np.zeros_like(adjacency)
    predecessors = np.zeros_like(adjacency, dtype = int)

    costs, predecessors = _floyd_warshall(
        adjacency,
        pivots_idx,
        costs,
        predecessors,
        maximum_path_cost,
    )

    return costs, predecessors

def recover_paths(graph, predecessors, **kwargs):

    node_to_idx = {k: idx for idx, k in enumerate(graph.nodes)}
    idx_to_node = {idx: k for idx, k in enumerate(graph.nodes)}

    # Recovering paths
    paths = {}

    for origin in graph.nodes:

        paths[origin] = {}

        for destination in graph.nodes:

            # print(origin, destination)

            path = _recover_path(
                predecessors, node_to_idx[origin], node_to_idx[destination]
                )

            paths[origin][destination] = [idx_to_node[n] for n in path]

    return paths

def recover_values(graph, paths, **kwargs):

    node_to_idx = {k: idx for idx, k in enumerate(graph.nodes)}

    adjacencies = kwargs.get('adjacencies', None)
    fields = kwargs.get('fields', ['distance'])

    if adjacencies is None:

        # Creating adjacency matrices
        adjacencies = (
            {f: nx.to_numpy_array(graph, weight = f, nonedge = np.inf) for f in fields}
            )

    # Recovering values
    values = {}

    for origin in graph.nodes:

        values[origin] = {}

        for destination in graph.nodes:

            path = [node_to_idx[n] for n in paths[origin][destination]]
            
            # print(path, [node_to_idx[n] for n in paths[destination][origin]])
            rpath = [node_to_idx[n] for n in paths[destination][origin]]

            v = (
                {f: _recover_path_costs(adjacencies[f], path) for f in fields}
                )

            vr = (
               {f: _recover_path_costs(adjacencies[f], rpath) for f in fields}
                )

            # print(v == vr, v, vr)

            values[origin][destination] = v

    arr = np.array([[v[fields[0]] for v in val.values()] for val in values.values()])

    return values

@jit(nopython = True, cache = True)
def _floyd_warshall(adjacency, pivots, costs, predecessors, max_cost):
    '''
    Implementation of Floyd Warshall algorithm
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
    '''

    n = len(adjacency)

    # Creating initial approximations
    for source in range(n):
        for target in range(n):

            # Initial assumption is that source is the direct predecessor to target
            # and that the cost is adjacency[source][target]. Non-edges should be
            # set to infinite cost for the algorithm to produce correct results
            costs[source][target] = adjacency[source][target]
            predecessors[source][target] = source

    # Updating approximations
    for pivot in pivots:
        for source in range(n):
            for target in range(n):

                tentative_cost = costs[source][pivot] + costs[pivot][target]

                # if source-pivot-target is lower cost than source-target then update
                if (tentative_cost < costs[source][target]) and (tentative_cost <= max_cost):

                    costs[source][target] = costs[source][pivot] + costs[pivot][target]
                    predecessors[source][target] = predecessors[pivot][target]

    return costs, predecessors

@jit(nopython = True, cache = True)
def _recover_path(predecessors, origin, destination):
    '''
    recovers paths by working backward from destination to origin
    '''

    max_iterations = len(predecessors)

    path = [destination]

    idx = 0

    while (origin != destination) and (idx <= max_iterations):

        # print(origin, destination)
        # print()

        destination = predecessors[origin][destination]
        path = [destination] + path

        # print(origin, destination)

        idx +=1

    return path

@jit(nopython = True, cache = True)
def _recover_path_costs(adjacency, path):
    '''
    Recovers costs for a path on an adjacency matrix
    '''

    cost = 0

    for idx in range(len(path) - 1):

        cost += adjacency[path[idx]][path[idx + 1]]

    return cost