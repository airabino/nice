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

        self.direct = kwargs.get('direct', {f: None for f in self.flows.keys()})

    def parameters(self, model):

        for destination, demand in self.flows.items():

            if self.direct[destination] is None:

                handle = f'{self.handle}::{destination}:direct'
                variable = pyomo.Param(
                    initialize = 0,
                    domain = pyomo.NonNegativeReals,
                    )
                setattr(model, handle, variable)
                self.handles.append(handle)

        return model

    def variables(self, model):

        for destination, demand in self.flows.items():

            if self.direct[destination] is not None:

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

            # print(destination, demand)

            direct = getattr(model, f"{self.handle}::{destination}:direct")
            # direct = 0

            volume = sum(
                path['object'].volume(model) \
                for path in self.paths[destination]
                )

            # print(volume)

            handle = f'{self.handle}:{destination}::volume_constraint'
            constraint = pyomo.Constraint(
                rule = demand * model.scale - volume - direct == 0
                )
            setattr(model, handle, constraint)

        return model

    def objective(self, model):

        cost = 0

        # if self.penalty is not None:

        direct = sum(
            [getattr(model, f"{self.handle}::{d}:direct") * c * model.penalty \
            for d, c in self.direct.items() if c is not None]
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

        results['direct'] = sum([v for k, v in results.items() if 'direct' in k])

        scale = pyomo.value(model.scale)

        results['total'] = sum([v * scale for k, v in self.flows.items()])

        results['direct_portion'] = results['direct'] / results['total']

        return results
