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

        self.penalty = kwargs.get('penalty', None)

        # print(self.penalty)

    def parameters(self, model):

        if self.penalty is None:

            # print('t')

            for destination, demand in self.flows.items():

                handle = f'{self.handle}::{destination}:direct'
                variable = pyomo.Param(
                    initialize = 0,
                    domain = pyomo.NonNegativeReals,
                    )
                setattr(model, handle, variable)
                self.handles.append(handle)

        return model

    def variables(self, model):

        if self.penalty is not None:

            for destination, demand in self.flows.items():

                handle = f'{self.handle}::{destination}:direct'
                variable = pyomo.Var(
                    initialize = 0,
                    domain = pyomo.NonNegativeReals,
                    )
                setattr(model, handle, variable)
                self.handles.append(handle)

        return model

    def constraints(self, model):

        for destination, demand in self.flows.items():

            direct = getattr(model, f"{self.handle}::{destination}:direct")

            volume = sum(
                path['object'].volume(model) \
                for path in self.paths[destination]
                )

            handle = f'{self.handle}:{destination}::volume_constraint'
            constraint = pyomo.Constraint(
                rule = demand * model.scale - volume - direct == 0
                )
            setattr(model, handle, constraint)

        return model

    def objective(self, model):

        cost = 0

        if self.penalty is not None:

            direct = sum(
                [getattr(model, f"{self.handle}::{d}:direct") * self.penalty[d] \
                for d in self.flows.keys()]
                )

            cost += direct

        return cost
    
    def results(self, model):

        results = {}

        for handle in self.handles:

            value = list(getattr(model, handle).extract_values().values())

            if len(value) == 1:

                value = value[0]

            results[handle.split('::')[1]] = value

        return results
