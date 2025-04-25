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

        self.power = kwargs.get('power', 80e3) # Charging power
        self.price = kwargs.get('price', .5 / 3.6e6)

        self.base_volume = kwargs.get('base_volume', 0)

        self.volumes = kwargs.get('volumes', [[0, sys.maxsize]])
        self.delays = kwargs.get('delays', [[0, 0]])

        self.delta_volume = np.diff(self.volumes, axis = 1)
        self.delta_delay = np.diff(self.delays, axis = 1)
        self.sizes, self.intervals = self.delta_volume.shape

        # print(self.delta_volume * 3600)

        self.size = kwargs.get('size', [1] + [0] * (self.sizes - 1))
        self.counts = kwargs.get('counts', np.arange(0, self.sizes) + 1)
        self.expenditures = kwargs.get('expenditures', np.arange(0, self.sizes) + 1)

    def charging_time(self, energy):

        return energy / self.power

    def sets(self, model):

        handle = f'{self.handle}::sizes'
        sizes = pyomo.Set(initialize = list(range(self.sizes)))
        setattr(model, handle, sizes)

        handle = f'{self.handle}::intervals'
        intervals = pyomo.Set(initialize = list(range(self.intervals)))
        setattr(model, handle, intervals)

        handle = f'{self.handle}::sizes_intervals'
        s = pyomo.Set(initialize = sizes * intervals)
        setattr(model, handle, s)

        return model

    def parameters(self, model):

        sizes = getattr(model, f'{self.handle}::sizes')

        if len(self.size) == 1:

            # Capacity intervals
            handle = f'{self.handle}::size'
            variable = pyomo.Param(
                sizes,
                initialize = [1] + [0] * (len(sizes) - 1),
                domain = pyomo.Binary,
                )
            setattr(model, handle, variable)
            self.handles.append(handle)

        # Baseline volume
        handle = f'{self.handle}::base_volume'
        variable = pyomo.Param(
            initialize = self.base_volume,
            domain = pyomo.NonNegativeReals,
            mutable = True,
            )
        setattr(model, handle, variable)
        self.handles.append(handle)

        return model

    def variables(self, model):

        # print('s')

        sizes = getattr(model, f'{self.handle}::sizes')
        intervals = getattr(model, f'{self.handle}::intervals')
        sizes_intervals = getattr(model, f'{self.handle}::sizes_intervals')

        if len(self.size) > 1:

            # Capacity intervals
            handle = f'{self.handle}::size'
            variable = pyomo.Var(
                sizes,
                initialize = [1] + [0] * (len(sizes) - 1),
                domain = pyomo.Binary,
                )
            setattr(model, handle, variable)
            self.handles.append(handle)

        # Capacity intervals
        handle = f'{self.handle}::usage'
        variable = pyomo.Var(
            sizes_intervals,
            initialize = 0,
            bounds = (0, 1),
            )
        setattr(model, handle, variable)
        self.handles.append(handle)

        # Capacity intervals
        # handle = f'{self.handle}::overflow'
        # variable = pyomo.Var(
        #     sizes,
        #     initialize = 0,
        #     domain = pyomo.NonNegativeReals,
        #     )
        # setattr(model, handle, variable)
        # self.handles.append(handle)

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

        sizes = getattr(model, f'{self.handle}::sizes')
        intervals = getattr(model, f'{self.handle}::intervals')

        usage = getattr(model, f'{self.handle}::usage')
        # overflow = getattr(model, f'{self.handle}::overflow')
        volume = getattr(model, f'{self.handle}::volume')
        delay = getattr(model, f'{self.handle}::delay')
        size = getattr(model, f'{self.handle}::size')

        base_volume = getattr(model, f'{self.handle}::base_volume')

        if len(self.size) > 1:

            sum_sizes = pyomo.quicksum(size[s] for s in sizes)

            handle = f'{self.handle}::size_unity_constraint'
            constraint = pyomo.Constraint(
                expr = sum_sizes == 1
                )
            setattr(model, handle, constraint)

        if self.path_handles:

            observed_volume = (sum(
                getattr(model, f'{p}::volume') for p in self.path_handles
                ) + base_volume) / model.duration

        else:

            observed_volume = base_volume / model.duration

        for i in sizes:

            usage_s = pyomo.quicksum(usage[i, j] for j in intervals)

            handle = f'{self.handle}::usage_size_constraint_{i}'
            constraint = pyomo.Constraint(
                expr = usage_s <= size[i] * self.intervals
                )
            setattr(model, handle, constraint)

        volume_sum = (
            pyomo.quicksum(
                usage[i , j] * self.delta_volume[i][j] \
                for i in sizes for j in intervals
                )
            # + pyomo.quicksum(overflow[i] for i in sizes)
            )

        delay_sum = (
            pyomo.quicksum(
                usage[i , j] * self.delta_delay[i][j] \
                for i in sizes for j in intervals
                )
            # + pyomo.quicksum(overflow[i] * self.delta_delay[i][-1] for i in sizes)
            )

        # Capacity
        handle = f'{self.handle}::flow_constraint'
        constraint = pyomo.Constraint(
            expr = volume_sum == observed_volume
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

    def expenditure(self, model):

        if len(self.size) == 1:

            expenditure = 0

        else:

            sizes = getattr(model, f'{self.handle}::sizes')
            size = getattr(model, f'{self.handle}::size')

            expenditure = pyomo.quicksum(size[s] * self.expenditures[s] for s in sizes)

        return expenditure

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

        # results['utilization'] = np.array(results['usage']).sum() / self.intervals
        results['utilization'] = results['volume'] / self.volumes.max()
        results['selection'] = self.counts[np.argmax(results['size'])]

        return results