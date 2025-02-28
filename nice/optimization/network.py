import sys
import time
# import pao

import numpy as np
import networkx as nx

import pyomo.environ as pyomo
import pyomo.opt as opt
import pyomo.util.model_size as model_size

# from pao.pyomo import *

from heapq import heappush, heappop
from itertools import count

from copy import deepcopy

from .base import Node, Edge
from .node import Place, Station
from .edge import Path
from .exceptions import *

from ..utilities import cprint
from ..graph import remove_self_edges

default_classes = ['Place', 'Station', 'Path']
base_classes = ['Node', 'Edge']

def level_graph(graph, origin, **kwargs):

    objective = kwargs.get('objective', 'objective')
    destinations = kwargs.get('destinations', [])

    _node = graph._node
    _adj = graph._adj

    costs = {} # dictionary of objective values for paths

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    heappush(heap, (0, next(c), origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost, _, source = heappop(heap)

        if source in costs:

            continue  # already searched this node.

        costs[source] = cost

        for target, edge in _adj[source].items():

            # Updating states for edge traversal
            cost_edge = edge.get(objective, 1)

            cost_target = cost + cost_edge

            # Updating the weighted cost for the path
            savings = cost_target < costs.get(target, np.inf)

            if savings:

                heappush(heap, (cost_target, next(c), target))

    destination_costs = [costs[d] for d in destinations]
    max_destination_cost = max(destination_costs)

    nodes = []
    edges = []

    for source, node in _node.items():

        nodes.append((source, node))

        if source in destinations:

            continue
        
        for target, edge in graph._adj[source].items():

            if costs[target] > costs[source] and costs[source] <= max_destination_cost:

                edges.append((source, target, edge))

    for destination in destinations:

        edge = _adj[origin][destination]

        edges.append((origin, destination, edge))

    level_graph = graph.__class__()
    level_graph.add_nodes_from(nodes)
    level_graph.add_edges_from(edges)

    return level_graph

class Network():

    def __init__(self, **kwargs):

        self.verbose = kwargs.get('verbose', False)

        # Demand scaling factor
        self.scale = kwargs.get('scale', 1)

        # Objective field
        self.objective = kwargs.get('objective', 'time')

        self.graph = nx.DiGraph()

    def size(self):

        return model_size.build_model_size_report(self.model)

    def solve(self, **kwargs):

        self.verbose = kwargs.get('verbose', self.verbose)
        tee = kwargs.get('tee', False)
        solver_kw = kwargs.get('solver', {'_name': 'glpk'})

        #Generating the solver object
        solver = opt.SolverFactory(**solver_kw)

        # self.model.dual = pyomo.Suffix(direction = pyomo.Suffix.IMPORT)

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

    def from_graph(self, graph):

        graph = deepcopy(graph)

        graph = remove_self_edges(graph)

        for source, node in graph._node.items():

            _class = node.pop('_class')

            self.add(_class, source, **node)

        for source, _adj in graph._adj.items():
            for target, edge in _adj.items():

                _class = edge.pop('_class')

                edge['source'] = source
                edge['target'] = target

                self.add(_class, f"{source}_{target}", **edge)

        return self

    def add(self, _class, handle, **kwargs):
        '''
        Adds and object to the network
        '''

        # Processing class
        if isinstance(_class, str):
            if _class in default_classes:

                _class = eval(_class)

            else:

                raise NICE_ClassNotFound

        # What is the base class of the object?
        _base = _class.__base__

        # print(handle, _base, Node)

        if _base is Node:

            # Add a node
            self.add_node(_class, handle, **kwargs)

        elif _base is Edge:

            source = kwargs.pop('source', None)
            target = kwargs.pop('target', None)

            # Add an edge
            self.add_edge(_class, handle, source, target, **kwargs)

        else:

            raise NICE_InvalidBaseClass

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

        # self.model.scale = pyomo.Var(
        #     initialize = 1, domain = (1, np.inf),
        #     )

        self.origins = [k for k, n in self.graph._node.items() if 'demand' in n]

        self.model.origins = pyomo.Set(
            initialize = self.origins
            )

        t0 = time.time()
        self.assign_edge_objects()
        cprint(f'Graph Transformed: {time.time() - t0}', self.verbose)

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
        self.build_objective()
        cprint(f'Objective Built: {time.time() - t0}', self.verbose)

    def build_objective(self):

        cost = 0

        for source, node in self.graph._node.items():

            cost += node['object'].objective(self.model)

            for target, edge in self.graph._adj[source].items():

                cost += edge['object'].objective(self.model)

        self.model.objective = pyomo.Objective(
            expr = cost, sense = pyomo.minimize
            )

        # self.model.objective = pyomo.Objective(
        #     expr = self.model.scale, sense = pyomo.maximize
        #     )

    def build_constraints(self):

        for source, node in self.graph._node.items():

            self.model = node['object'].constraints(self.model)

            for target, edge in self.graph._adj[source].items():

                self.model = edge['object'].constraints(self.model)

    def build_variables(self):

        for source, node in self.graph._node.items():

            self.model = node['object'].variables(self.model)

            for target, edge in self.graph._adj[source].items():

                self.model = edge['object'].variables(self.model)

    def build_parameters(self):

        for source, node in self.graph._node.items():

            self.model = node['object'].parameters(self.model)

            for target, edge in self.graph._adj[source].items():

                self.model = edge['object'].parameters(self.model)

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