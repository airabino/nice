import sys
import time

import numpy as np
import pyomo.environ as pyomo
import networkx as nx

from .base import Object

class Path(Object):

    def __init__(self, handle, **kwargs):

        super().__init__(handle, **kwargs)

        self.nodes = kwargs.get('nodes', [])
        self.edges = kwargs.get('edges', [])
        
        self.capacity = kwargs.get('capacity', 1)
        self.initial = kwargs.get('initial', self.capacity)

        # print('c', self.capacity)

        self.penalty = kwargs.get('penalty', 1)

    def parameters(self, model):

        handle = f'{self.handle}::nodes'
        s = pyomo.Set(initialize = list(range(len(self.nodes))))
        setattr(model, handle, s)

        return model

    def variables(self, model):

        # Volume on path
        handle = f"{self.handle}::volume"
        self.handles.append(handle)
        setattr(
            model, handle,
            pyomo.Var(
                initialize = 0,
                within = pyomo.NonNegativeReals
                ),
            )

        # Level of charge at nodes
        # for node in self.nodes:

        nodes = getattr(model, f'{self.handle}::nodes')

        handle = f"{self.handle}::level"
        self.handles.append(handle)
        setattr(
            model, handle,
            pyomo.Var(
                nodes,
                initialize = [0] * len(nodes),
                within = pyomo.NonNegativeReals
                ),
            )

        return model

    def constraints(self, model):

        volume = getattr(model, f'{self.handle}::volume')
        nodes = getattr(model, f'{self.handle}::nodes')
        level = getattr(model, f'{self.handle}::level')
        # consume = getattr(model, f'{self.handle}::consume')

        # nodes.pprint()
        # print(len(self.edges))

        # State of Charge
        
        # print()
        def level_rule(m, n):

            if n == 0:

                rule = volume * self.initial == level[n]

            else:

                rule = level[n] == (
                    level[n - 1] -
                    volume * self.edges[n - 1]['object'].energy(model) +
                    self.nodes[n]['object'].energy(model, self.handle)
                    )

            return rule

        setattr(
            model, f"{self.handle}::level_constraint",
            pyomo.Constraint(
                nodes,
                rule = lambda m, n: level_rule(m, n),
                )
            )

        def level_bounds_rule(m, n):

            rule = volume * self.capacity >= level[n]

            return rule

        setattr(
            model, f"{self.handle}::level_bounds_constraint",
            pyomo.Constraint(
                nodes,
                rule = lambda m, n: level_bounds_rule(m, n),
                )
            )

        return model

    def volume(self, model):

        volume = getattr(model, f'{self.handle}::volume')

        return volume

    def objective(self, model):

        edge_cost = sum(e['object'].objective(model) for e in self.edges)

        volume = getattr(model, f'{self.handle}::volume')

        cost = volume * edge_cost

        return cost

    def results(self, model):

        results = {}

        for handle in self.handles:

            value = list(getattr(model, handle).extract_values().values())

            if len(value) == 1:

                value = value[0]

            else:

                value = np.array(value) / self.capacity

            results[handle.split('::')[1]] = value

        # results['total_flow'] = sum([v for k, v in results.items() if 'flow:' in k])

        return results