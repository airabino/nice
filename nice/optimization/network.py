import sys
import time
# import pao

import numpy as np
import networkx as nx

import pyomo.environ as pyomo
import pyomo.opt as opt
import pyomo.util.model_size as model_size

from heapq import heappush, heappop
from itertools import count

from copy import deepcopy

from ..utilities import cprint, nested_add
from ..graph import remove_self_edges, level_graph, k_shortest_paths
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

def get_paths(graph, terminals = None, k = None, weight = None):

    paths = []

    if terminals is None:

        terminals = list(graph.nodes())

    for origin in terminals:

        destinations = set(terminals) - set([origin])

        lg = level_graph(graph, origin, destinations, weight = weight)

        for destination in destinations:

            ksp = k_shortest_paths(lg, origin, destination, k = k, weight = weight)

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

def solution_atlas(solution, atlas, fields = []):

    graph = atlas.copy()

    for source, node in solution._node.items():

        graph._node[source] = node

    for source, _adj in solution._adj.items():
        for target, val in _adj.items():

            path = val['path']

            for idx in range(len(path) - 1):

                edge = graph._adj[path[idx]][path[idx + 1]]

                for field in fields:

                    if field in edge:

                        edge[field] += val[field]

                    else:

                        edge[field] = val[field]

    return graph

class Network():

    def __init__(self, **kwargs):

        self.graph = nx.DiGraph()
        self.paths = []

        self.verbose = kwargs.get('verbose', False)

        # Demand scaling factor
        self.scale = kwargs.get('scale', 1)

        self.expenditure = kwargs.get('expenditure', np.inf)

        # Objective field
        self.objective = kwargs.get('objective', 'time')

    def size(self):

        return model_size.build_model_size_report(self.model)

    def solve(self, **kwargs):

        self.verbose = kwargs.get('verbose', self.verbose)
        tee = kwargs.get('tee', False)
        solver_kw = kwargs.get('solver', {'_name': 'glpk'})

        #Generating the solver object
        solver = opt.SolverFactory(**solver_kw)

        # Building and solving as a linear problem
        t0 = time.time()
        self.result = solver.solve(self.model, tee = tee)
        cprint(f'Problem Solved: {time.time() - t0}', self.verbose)

        # Making solution dictionary
        t0 = time.time()
        self.collect_results()
        cprint(f'Results Collected: {time.time() - t0}', self.verbose)

    def collect_results(self):

        nodes = []
        edges = []

        for source, node in self.graph._node.items():

            node_results = node['object'].results(self.model)

            nodes.append((source, {**node, **node_results}))

            for target, edge in self.graph._adj[source].items():

                edge_results = edge['object'].results(self.model)

                edges.append((source, target, {**edge, **edge_results}))


        self.solution = nx.DiGraph()
        self.solution.add_nodes_from(nodes)
        self.solution.add_edges_from(edges)

        for path in self.paths:

            path['results'] = path['object'].results(self.model)

    def from_graph(self, graph, paths = []):

        graph = deepcopy(graph)
        paths = deepcopy(paths)

        graph = remove_self_edges(graph)

        for source, node in graph._node.items():

            _class = node.pop('_class')

            self.add_node(_class, source, **node)

        for source, _adj in graph._adj.items():
            for target, edge in _adj.items():

                _class = edge.pop('_class')

                self.add_edge(_class, f"{source}_{target}", source, target, **edge)

        for path in paths:

            p = path['path']
            path['nodes'] = []
            path['edges'] = []

            handle = f"{path['origin']}_{path['destination']}_{path['index']}"

            for idx in range(1, len(p)):

                source = p[idx - 1]
                target = p[idx]

                path['nodes'].append(self.graph._node[source])
                path['edges'].append(self.graph._adj[source][target])

                self.graph._node[source]['object'].path_handles.append(handle)
                self.graph._adj[source][target]['object'].path_handles.append(handle)

            _class = path.pop('_class')

            pointer = {**path, 'object': _class(handle, **path)}

            self.paths.append(pointer)
            self.graph._node[path['origin']]['object'].paths[path['destination']].append(
                pointer
                )

        return self

    def add_node(self, _class, handle, **kwargs):

        self.graph.add_node(handle, object = _class(handle, **kwargs), **kwargs)

    def add_edge(self, _class, handle, source, target, **kwargs):

        kwargs['unit_cost'] = kwargs.get(self.objective, 1)

        edge_obj = _class(handle, **kwargs)

        self.graph.add_edge(source, target, object = edge_obj, **kwargs)

    def build(self):

        self.model = pyomo.ConcreteModel()

        self.model.scale = pyomo.Param(
            initialize = self.scale, mutable = True
            )

        self.model.expenditure = pyomo.Param(
            initialize = self.expenditure, mutable = True
            )

        t0 = time.time()
        self.build_sets()
        cprint(f'Sets Built: {time.time() - t0}', self.verbose)

        t0 = time.time()
        self.build_parameters()
        cprint(f'Parameters Built: {time.time() - t0}', self.verbose)

        t0 = time.time()
        self.build_variables()
        cprint(f'Variables Built: {time.time() - t0}', self.verbose)

        t0 = time.time()
        self.build_constraints()
        cprint(f'Constraints Built: {time.time() - t0}', self.verbose)

        t0 = time.time()
        self.build_expenditure()
        cprint(f'Expenditure Built: {time.time() - t0}', self.verbose)

        t0 = time.time()
        self.build_objective()
        cprint(f'Objective Built: {time.time() - t0}', self.verbose)

    def build_sets(self):

        for source, node in self.graph._node.items():

            self.model = node['object'].sets(self.model)

            for target, edge in self.graph._adj[source].items():

                self.model = edge['object'].sets(self.model)

        for path in self.paths:

            self.model = path['object'].sets(self.model)

    def build_objective(self):

        cost = 0

        for source, node in self.graph._node.items():

            cost += node['object'].objective(self.model)

            for target, edge in self.graph._adj[source].items():

                cost += edge['object'].objective(self.model)

        for path in self.paths:

            cost += path['object'].objective(self.model)

        self.model.objective = pyomo.Objective(
            expr = cost, sense = pyomo.minimize
            )

    def build_expenditure(self):

        expenditure = 0

        for source, node in self.graph._node.items():

            expenditure += node['object'].expenditure(self.model)

            for target, edge in self.graph._adj[source].items():

                expenditure += edge['object'].expenditure(self.model)

        for path in self.paths:

            expenditure += path['object'].expenditure(self.model)

        # print(expenditure)

        self.model.requirement_constraint = pyomo.Constraint(
            rule = expenditure <= self.model.expenditure
            )

    def build_constraints(self):

        for source, node in self.graph._node.items():

            self.model = node['object'].constraints(self.model)

            for target, edge in self.graph._adj[source].items():

                self.model = edge['object'].constraints(self.model)

        for path in self.paths:

            self.model = path['object'].constraints(self.model)

    def build_variables(self):

        for source, node in self.graph._node.items():

            self.model = node['object'].variables(self.model)

            for target, edge in self.graph._adj[source].items():

                self.model = edge['object'].variables(self.model)

        for path in self.paths:

            self.model = path['object'].variables(self.model)

    def build_parameters(self):

        for source, node in self.graph._node.items():

            self.model = node['object'].parameters(self.model)

            for target, edge in self.graph._adj[source].items():

                self.model = edge['object'].parameters(self.model)

        for path in self.paths:

            self.model = path['object'].parameters(self.model)

    def assign_edge_objects(self):

        for source, node in self.graph._node.items():

            node['object'].successors = {o: [] for o in self.origins}
            node['object'].predecessors = {o: [] for o in self.origins}

        for origin in self.origins:

            destinations = set(self.origins) - set([origin])

            lg = level_graph(
                self.graph, origin,
                destinations = destinations,
                objective = self.objective,
                )

            for source, _adj in lg._adj.items():

                source_node = self.graph._node[source]

                for target, edge in _adj.items():

                    target_node = self.graph._node[target]

                    source_node['object'].successors[origin].append(edge)
                    target_node['object'].predecessors[origin].append(edge)