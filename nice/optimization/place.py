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

        self.mode_switch = kwargs.get('mode_switch', {f: None for f in self.flows.keys()})

    def parameters(self, model):

        for destination, demand in self.flows.items():

            if self.mode_switch[destination] is None:

                handle = f'{self.handle}::{destination}:mode_switch'
                variable = pyomo.Param(
                    initialize = 0,
                    domain = pyomo.NonNegativeReals,
                    )
                setattr(model, handle, variable)
                self.handles.append(handle)

        return model

    def variables(self, model):

        for destination, demand in self.flows.items():

            if self.mode_switch[destination] is not None:

                handle = f'{self.handle}::{destination}:mode_switch'
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

            mode_switch = getattr(model, f"{self.handle}::{destination}:mode_switch")

            volume = sum(
                path['object'].volume(model) \
                for path in self.paths[destination]
                )

            # print(volume)

            handle = f'{self.handle}:{destination}::volume_constraint'
            constraint = pyomo.Constraint(
                # rule = demand * model.scale - volume == 0
                rule = demand * model.scale - volume - mode_switch == 0
                )
            setattr(model, handle, constraint)

        return model

    def objective(self, model):

        mode_switch = sum(
            [getattr(model, f"{self.handle}::{d}:mode_switch") * c * model.penalty \
            for d, c in self.mode_switch.items() if c is not None]
            )

        cost = mode_switch

        return cost
    
    def results(self, model):

        results = {}

        for handle in self.handles:

            value = list(getattr(model, handle).extract_values().values())

            if len(value) == 1:

                value = value[0]

            results[handle.split('::')[1]] = value

        results['mode_switch'] = sum([v for k, v in results.items() if 'mode_switch' in k])

        scale = pyomo.value(model.scale)

        results['total'] = sum([v * scale for k, v in self.flows.items()])

        if results['total'] == 0:

            results['mode_switch_portion'] = 0

        else:

            results['mode_switch_portion'] = results['mode_switch'] / results['total']

        return results
