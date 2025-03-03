import sys
import time

import numpy as np
import pyomo.environ as pyomo
import networkx as nx

from ..base.node import Node

class Place(Node):

    def __init__(self, handle, **kwargs):

        super().__init__(handle, **kwargs)

        self.predecessors = kwargs.get('predecessors', {})
        self.successors = kwargs.get('successors', {})

        # Demand is the amount of traffic a place pulls from other places
        # {origin: volume}
        self.demand = kwargs.get('demand', {})

    def constraints(self, model):

        # for each origin, the flow into the place has to sum to that demand
        for origin in model.origins:

            volume = self.demand.get(origin, 0)

            in_flow = sum(
                predecessor['object'].volume(model, origin) \
                for predecessor in self.predecessors[origin]
                )

            handle = f'{self.handle}:{origin}::in_flow_constraint'
            constraint = pyomo.Constraint(
                rule = volume * model.scale - in_flow == 0
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
