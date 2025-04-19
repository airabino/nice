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
        
        # self.capacity = kwargs.get('capacity', 1)
        # self.initial = kwargs.get('initial', self.capacity)

        # print('c', self.capacity)

        # self.penalty = kwargs.get('penalty', 1)

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

        return model

    def volume(self, model):

        volume = getattr(model, f'{self.handle}::volume')

        return volume

    def objective(self, model):

        edge_cost = sum(e['object'].cost for e in self.edges)

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

        return results