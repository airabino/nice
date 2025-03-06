import sys
import time

import numpy as np
import pyomo.environ as pyomo
import networkx as nx

from ..base.edge import Edge

class Path(Edge):

    def __init__(self, handle, **kwargs):

        super().__init__(handle, **kwargs)

        self.unit_cost = kwargs.get('unit_cost', 1)
        self.capacity = kwargs.get('capacity', np.inf)

        # print(self.unit_cost)

    def variables(self, model):

        # Transmit flow for each origin
        for origin in model.origins:

            handle = f"{self.handle}::flow:{origin}"
            self.handles.append(handle)
            setattr(
                model, handle,
                pyomo.Var(
                    initialize = 0,
                    within = pyomo.NonNegativeReals
                    ),
                )

        return model

    def constraints(self, model):

        if self.capacity < np.inf:

            # print(self.handle)

            volume = sum(
                getattr(model, f'{self.handle}::flow:{o}') for o in model.origins
                )

            handle = f'{self.handle}::capacity_constraint'
            constraint = pyomo.Constraint(
                rule = volume <= self.capacity
                )
            setattr(model, handle, constraint)

        return model

    def volume(self, model, origin):

        flow = getattr(model, f"{self.handle}::flow:{origin}")

        return flow

    def objective(self, model):

        volume = sum(
            getattr(model, f'{self.handle}::flow:{o}') for o in model.origins
            )

        cost = volume * self.unit_cost

        return cost

    def results(self, model):

        results = {}

        for handle in self.handles:

            value = list(getattr(model, handle).extract_values().values())

            if len(value) == 1:

                value = value[0]

            results[handle.split('::')[1]] = value

        results['total_flow'] = sum([v for k, v in results.items() if 'flow:' in k])

        return results