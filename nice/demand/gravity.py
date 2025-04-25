import time
import numpy as np
import networkx as nx

from scipy.stats import rv_histogram

from ..graph import path_cost, dijkstra

def friction_exponential(distance, a = 0.00890009, b = -0.00686443, c = 1 / 1609):

    return a * np.exp(b * c * distance)

def charge_time(costs, power = 6600):

    # costs['time'] += costs['time'] + costs['energy'] / power

    return costs['time'] + costs['energy'] / power

def within_range(costs, capacity = 75 * 3.6e6):

    return costs['energy'] <= capacity

def demand(graph, places, **kwargs):

    production = kwargs.get('production', 'production')
    routing_weight = kwargs.get('routing_weight', 'time')
    demand_weight = kwargs.get('demand_weight', 'distance')
    friction_function = kwargs.get('friction_function', friction_exponential)
    penalty_function = kwargs.get('penalty_function', charge_time)
    remove_function = kwargs.get('remove_function', within_range)

    sum_demand = 0
    min_costs = {p: {p: 0 for p in places} for p in places}
    max_costs = {p: {p: 0 for p in places} for p in places}
    friction = {p: {p: 0 for p in places} for p in places}
    trips = {p: {p: 0 for p in places} for p in places}

    for origin in places:

        destinations = set(places) - set([origin])

        _, values, paths = dijkstra(
            graph, [origin],
            objective = routing_weight,
            fields = [routing_weight, demand_weight, 'energy'],
            )

        for destination in destinations:

            path = paths[destination]
            path_costs = values[destination]

            min_costs[origin][destination] = path_costs
            max_costs[origin][destination] = penalty_function(path_costs)
            friction[origin][destination] = friction_function(
                path_costs[demand_weight]
                )

            sum_demand += (
                graph._node[origin][production] *
                friction[origin][destination]
            )

    trips = {p: {p: 0 for p in places} for p in places}

    sum_trips = 0

    for origin in places:

        destinations = set(places) - set([origin])
        
        for destination in destinations:

            trips[origin][destination] = (
                graph._node[origin][production] * 
                friction[origin][destination] *
                graph._node[destination][production]
            )

            sum_trips += trips[origin][destination]

    for o in places:

        destinations = set(places) - set([o])

        graph._node[o]['flows'] = {}
        graph._node[o]['free_flow'] = {}
        graph._node[o]['mode_switch'] = {}

        for d in destinations:

            if remove_function(min_costs[o][d]):

                continue

            graph._node[o]['flows'][d] = trips[o][d] / sum_trips
            graph._node[o]['free_flow'][d] = min_costs[o][d][routing_weight]
            graph._node[o]['mode_switch'][d] = max_costs[o][d]
    
    return graph