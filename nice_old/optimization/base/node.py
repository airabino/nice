import pyomo.environ as pyomo

class Node:

    __base__ = 'Node'

    def __init__(self, handle, **kwargs):

        self.handle = handle
        self.handles = []

    def parameters(self, model):

        return model

    def variables(self, model):

        return model

    def constraints(self, model):

        return model

    def objective(self, model):
        """Base objective function returns zero cost"""

        return 0.  # Default to no cost for nodes

    def results(self, model):

        results = {}

        for handle in self.handles:

            value = list(getattr(model, handle).extract_values().values())
            results[handle] = value

        return results 