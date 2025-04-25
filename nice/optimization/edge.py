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

        self.energy = kwargs.get('energy', 0)
        self.cost = kwargs.get('cost', 0)
    
    def results(self, model):

        results = {}

        for p in self.path_handles:

            handle = f"{p}::volume"

            value = list(getattr(model, handle).extract_values().values())

            if len(value) == 1:

                value = value[0]

            results[handle] = value

        results['volume'] = sum(results.values())

        return results
