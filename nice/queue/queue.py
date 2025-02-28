import time
import numpy as np

from scipy.special import factorial
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

def mmc_queue(arrival_rate, service_rate, servicers, cutoff = np.inf):
    '''
    Computes TOTAL waiting time in queue (L_q ** 2 / lambda)
    '''

    arrival_rate = np.atleast_1d(arrival_rate)
    service_rate = np.atleast_1d(service_rate)

    rho = arrival_rate / (service_rate * servicers)

    probability_empty_denominator = 0

    for k in range(servicers):

        probability_empty_denominator += (servicers * rho) ** k / factorial(k)

    probability_empty_denominator += (
        (servicers * rho) ** servicers / factorial(servicers) / (1 - rho)
        )

    probability_empty = 1 / probability_empty_denominator

    # waiting_time = (
    #     probability_empty * rho * (servicers * rho) ** servicers /
    #     (arrival_rate * (1 - rho) ** 2 * factorial(servicers))
    #     )

    mean_queue_length = (
        probability_empty * rho * (servicers * rho) ** servicers /
        ((1 - rho) ** 2 * factorial(servicers))
        )

    waiting_time = mean_queue_length ** 2 / arrival_rate

    waiting_time[rho == 0] = 0
    waiting_time[rho >= 1] = cutoff
    waiting_time[np.isnan(rho)] = cutoff

    return waiting_time

def probability_empty(arrival_rate, service_rate, servicers):

    arrival_rate = np.atleast_1d(arrival_rate)
    service_rate = np.atleast_1d(service_rate)

    rho = arrival_rate / (service_rate * servicers)

    probability_empty_denominator = 0

    for k in range(servicers):

        probability_empty_denominator += (servicers * rho) ** k / factorial(k)

    probability_empty_denominator += (
        (servicers * rho) ** servicers / factorial(servicers) / (1 - rho)
        )

    probability_empty = 1 / probability_empty_denominator

    return probability_empty

class Queue():

    def __init__(self, **kwargs):

        self.rho = kwargs.get('rho', np.linspace(0, .99, 100))
        self.m = kwargs.get('m', 1)
        self.c = kwargs.get('c', list(range(1, 101)))
        self.cutoff = kwargs.get('cutoff', np.inf)
        self.bounds = kwargs.get('bounds', [(0, 3)] * 3)
        self.initial_guess = kwargs.get('initial_guess', [1.25, 1, .15])

        self.build()

    def interpolate(self, rho, c):

        result = self.interpolator((c, rho))
        result[result > self.cutoff] = self.cutoff

        return result

    def approximation_function(self, beta, rho, c):

        return beta[0] / (1 - rho ** (beta[1] * c + beta[2])) - beta[0]

    def difference(self, beta, rho, c, actual):

        approximation = self.approximation_function(beta, rho, c)

        return ((actual - approximation.T) ** 2).sum()

    def fit(self):

        rho_g, c_g = np.meshgrid(self.rho, self.c, indexing = 'ij')

        result = minimize(
            lambda beta: self.difference(beta, rho_g, c_g, self.waiting_times),
            self.initial_guess, bounds = self.bounds
            )

        print(result)

        self.beta = result.x

    def approximate(self, rho, c):

        return self.approximation_function(self.beta, rho, c)

    def build(self):

        self.waiting_times = np.zeros((len(self.c), len(self.rho)))

        for idx, c in enumerate(self.c):

            arrival_rate = self.rho * c * self.m

            self.waiting_times[idx] = mmc_queue(
                arrival_rate, self.m, c, cutoff = self.cutoff,
                )

        self.interpolator = RegularGridInterpolator(
            (self.c, self.rho), self.waiting_times,
            bounds_error = False, fill_value = np.inf,
            )