import sys
import time

import numpy as np
import pyomo.environ as pyomo
import networkx as nx

from .base import Object

class Place(Object):

    def __init__(self, handle, **kwargs):

        super().__init__(handle, **kwargs)

        self.path_handles = kwargs.get('path_handles', [])

        self.flows = kwargs.get('flows', {})
        self.paths = {f: [] for f in self.flows.keys()}

    def constraints(self, model):



        for destination, demand in self.flows.items():

            # print(self.paths[destination])

            volume = sum(
                path['object'].volume(model) \
                for path in self.paths[destination]
                )

            handle = f'{self.handle}:{destination}::volume_constraint'
            constraint = pyomo.Constraint(
                rule = demand * model.scale - volume == 0
                )
            setattr(model, handle, constraint)

        return model
    
    def results(self, model):

        results = {}

        for handle in self.handles:

            value = list(getattr(model, handle).extract_values().values())

            if len(value) == 1:

                value = value[0]

            results[handle.split('::')[1]] = value

        return results
