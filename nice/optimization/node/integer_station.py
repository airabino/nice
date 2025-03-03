import sys
import time

import numpy as np
import pyomo.environ as pyomo
import networkx as nx

from ..base.node import Node

class Station(Node):

    def __init__(self, handle, **kwargs):

        super().__init__(handle, **kwargs)

        self.predecessors = kwargs.get('predecessors', {})
        self.successors = kwargs.get('successors', {})

        self.volumes = kwargs.get('volumes', [0, sys.maxsize])
        self.delays = kwargs.get('delays', [0, 0])
        self.delta_volume = np.diff(self.volumes)
        self.delta_delay = np.diff(self.delays)
        self.intervals = len(self.delta_volume)

        self.power = kwargs.get('rate', 80e3) # Charging power

    def parameters(self, model):

        # print(self.handle, 'p', self.predecessors.keys())
        # print(self.handle, 's', self.successors.keys())

        handle = f'{self.handle}::intervals'
        s = pyomo.Set(initialize = list(range(self.intervals)))
        setattr(model, handle, s)

        return model

    def variables(self, model):

        intervals = getattr(model, f'{self.handle}::intervals')

        # Capacity intervals
        handle = f'{self.handle}::usage'
        variable = pyomo.Var(
            intervals,
            initialize = [0] * len(intervals),
            bounds = (0, 1),
            )
        setattr(model, handle, variable)
        self.handles.append(handle)

        # Tracking values
        handle = f'{self.handle}::volume'
        variable = pyomo.Var(
            initialize = 0,
            )
        setattr(model, handle, variable)
        self.handles.append(handle)

        handle = f'{self.handle}::delay'
        variable = pyomo.Var(
            initialize = 0,
            )
        setattr(model, handle, variable)
        self.handles.append(handle)

        return model

    def constraints(self, model):

        gross_flow = 0

        # Net flow must be zero for each origin
        for origin in model.origins:

            if (self.successors[origin] == []) and (self.predecessors[origin] == []):

                continue

            out_flow = sum(
                successor['object'].volume(model, origin) \
                for successor in self.successors[origin]
                )

            in_flow = sum(
                predecessor['object'].volume(model, origin) \
                for predecessor in self.predecessors[origin]
                )

            # print(self.handle, in_flow)

            gross_flow += in_flow

            handle = f'{self.handle}:{origin}::net_flow_constraint'
            constraint = pyomo.Constraint(
                expr = out_flow == in_flow
                )
            setattr(model, handle, constraint)

        usage = getattr(model, f'{self.handle}::usage')
        intervals = getattr(model, f'{self.handle}::intervals')
        volume = getattr(model, f'{self.handle}::volume')
        delay = getattr(model, f'{self.handle}::delay')

        volume_sum = pyomo.quicksum(usage[i] * self.delta_volume[i] for i in intervals)
        delay_sum = pyomo.quicksum(usage[i] * self.delta_delay[i] for i in intervals)

        # print(gross_flow)

        # Capacity
        handle = f'{self.handle}::flow_constraint'
        constraint = pyomo.Constraint(
            expr = volume_sum == gross_flow
            )
        setattr(model, handle, constraint)

        # Volume tracking
        handle = f'{self.handle}::volume_tracking_constraint'
        constraint = pyomo.Constraint(
            expr = volume == volume_sum
            )
        setattr(model, handle, constraint)

        # Delay tracking
        handle = f'{self.handle}::delay_tracking_constraint'
        constraint = pyomo.Constraint(
            expr = delay == delay_sum
            )
        setattr(model, handle, constraint)

        return model

    def objective(self, model):

        delay = getattr(model, f'{self.handle}::delay')

        return delay
    
    def results(self, model):

        results = {}

        for handle in self.handles:

            value = list(getattr(model, handle).extract_values().values())

            if len(value) == 1:

                value = value[0]

            results[handle.split('::')[1]] = value

        results['utilization'] = sum(results['usage']) / len(results['usage'])

        return results
