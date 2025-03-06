import sys
import time

import numpy as np
import pyomo.environ as pyomo
import networkx as nx

from .base import Object

class Station(Object):

    def __init__(self, handle, **kwargs):

        super().__init__(handle, **kwargs)

        self.path_handles = kwargs.get('path_handles', [])

        self.capacity = kwargs.get('capacity', sys.maxsize)
        self.power = kwargs.get('power', 1) # Charging power
        self.cost = kwargs.get('cost', 1 / self.power)

    def variables(self, model):

        for p in self.path_handles:

            handle = f'{self.handle}::{p}:energy'
            variable = pyomo.Var(
                initialize = 0,
                domain = pyomo.NonNegativeReals,
                )
            setattr(model, handle, variable)
            self.handles.append(handle)

        return model

    def constraints(self, model):

        # print(len(self.path_handles))

        if self.path_handles:

            energy = sum(
                getattr(model, f'{self.handle}::{p}:energy') for p in self.path_handles
                )

            # Capacity
            handle = f'{self.handle}::energy_constraint'
            constraint = pyomo.Constraint(
                expr = energy <= self.capacity
                )
            setattr(model, handle, constraint)

        return model

    def energy(self, model, path = None):

        if path is None:

            energy = sum(
                getattr(model, f'{self.handle}::{p}:energy') for p in self.path_handles
                )

        else:

            energy = getattr(model, f'{self.handle}::{path}:energy')

        return energy

    def objective(self, model):

        energy = self.energy(model)

        cost = energy * self.cost

        return cost
    
    def results(self, model):

        results = {}

        for handle in self.handles:

            value = list(getattr(model, handle).extract_values().values())

            if len(value) == 1:

                value = value[0]

            results[handle.split('::')[1]] = value

        results['energy'] = sum(results.values())

        return results

# class Station_c(Node):

#     def __init__(self, handle, **kwargs):

#         super().__init__(handle, **kwargs)

#         self.paths = kwargs.get('paths', [])

#         self.volumes = kwargs.get('volumes', [0, sys.maxsize])
#         self.delays = kwargs.get('delays', [0, 0])
#         self.delta_volume = np.diff(self.volumes)
#         self.delta_delay = np.diff(self.delays)
#         self.intervals = len(self.delta_volume)

#         self.power = kwargs.get('rate', 80e3) # Charging power

#     def parameters(self, model):

#         handle = f'{self.handle}::intervals'
#         s = pyomo.Set(initialize = list(range(self.intervals)))
#         setattr(model, handle, s)

#         return model

#     def variables(self, model):

#         for path in self.paths:

#             handle = f'{self.handle}:{path['object'].handle}::energy'
#             variable = pyomo.Var(
#                 initialize = 0,
#                 domain = pyomo.NonNegativeReals,
#                 )
#             setattr(model, handle, variable)
#             self.handles.append(handle)

#         intervals = getattr(model, f'{self.handle}::intervals')

#         # Capacity intervals
#         handle = f'{self.handle}::usage'
#         variable = pyomo.Var(
#             intervals,
#             initialize = [0] * len(intervals),
#             bounds = (0, 1),
#             )
#         setattr(model, handle, variable)
#         self.handles.append(handle)

#         # Tracking values
#         handle = f'{self.handle}::volume'
#         variable = pyomo.Var(
#             initialize = 0,
#             )
#         setattr(model, handle, variable)
#         self.handles.append(handle)

#         handle = f'{self.handle}::delay'
#         variable = pyomo.Var(
#             initialize = 0,
#             )
#         setattr(model, handle, variable)
#         self.handles.append(handle)

#         return model

#     def constraints(self, model):

#         gross_flow = sum(
#             getattr(model, f'{self.handle}:{path}::energy') for path in self.paths
#             )

#         usage = getattr(model, f'{self.handle}::usage')
#         intervals = getattr(model, f'{self.handle}::intervals')
#         volume = getattr(model, f'{self.handle}::volume')
#         delay = getattr(model, f'{self.handle}::delay')

#         volume_sum = pyomo.quicksum(usage[i] * self.delta_volume[i] for i in intervals)
#         delay_sum = pyomo.quicksum(usage[i] * self.delta_delay[i] for i in intervals)

#         # print(gross_flow)

#         # Capacity
#         handle = f'{self.handle}::flow_constraint'
#         constraint = pyomo.Constraint(
#             expr = volume_sum == gross_flow
#             )
#         setattr(model, handle, constraint)

#         # Volume tracking
#         handle = f'{self.handle}::volume_tracking_constraint'
#         constraint = pyomo.Constraint(
#             expr = volume == volume_sum
#             )
#         setattr(model, handle, constraint)

#         # Delay tracking
#         handle = f'{self.handle}::delay_tracking_constraint'
#         constraint = pyomo.Constraint(
#             expr = delay == delay_sum
#             )
#         setattr(model, handle, constraint)

#         return model

#     def objective(self, model):

#         delay = getattr(model, f'{self.handle}::delay')

#         return delay
    
#     def results(self, model):

#         results = {}

#         for handle in self.handles:

#             value = list(getattr(model, handle).extract_values().values())

#             if len(value) == 1:

#                 value = value[0]

#             results[handle.split('::')[1]] = value

#         results['utilization'] = sum(results['usage']) / len(results['usage'])

#         return results