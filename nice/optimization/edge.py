import sys
import time

import numpy as np
import pyomo.environ as pyomo
import networkx as nx

from .base import Object

class Edge(Object):

    def __init__(self, handle, **kwargs):

        super().__init__(handle, **kwargs)

        self.path_handles = kwargs.get('path_handles', [])

        self.consumption = kwargs.get('consumption', 0)
        self.cost = kwargs.get('cost', 0)

        # print(self.energy)

    def energy(self, model):

        return self.consumption

    def objective(self, model):

        return self.cost
    
    def results(self, model):

        results = {}

        for p in self.path_handles:

            handle = f"{p}::volume"

            value = list(getattr(model, handle).extract_values().values())

            if len(value) == 1:

                value = value[0]

            results[handle] = value

        # print(results)

        results['volume'] = sum(results.values())

        return results
