import time
import numpy as np

from heapq import heappush, heappop

from scipy.special import factorial
from scipy.stats import binom
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

def mmc_sim(l, m, c, n):

    arrivals = np.cumsum(l.rvs(size = n))
    services = m.rvs(size = n)

    population = []
    queue = []
    service = []
    # count = count()

    for idx in range(n):

        heappush(population, (arrivals[idx], services[idx], idx))
    
    occupied = 0
    arrival_times = []
    start_times = []
    finish_times = []

    sim_time = 0

    while population:

        arrival_time, service_time, idx = heappop(population)

        if occupied > 0:

            while service[0][0] <= arrival_time:

                _ = heappop(service)
                occupied -= 1

                if not service:

                    break

        if occupied < c:

            start_time = arrival_time
            finish_time = start_time + service_time
            occupied += 1

            heappush(service, (finish_time, idx))

        else:

            start_time, _ = heappop(service)
            finish_time = start_time + service_time

            heappush(service, (finish_time, idx))

        



        arrival_times.append(arrival_time)
        start_times.append(start_time)
        finish_times.append(finish_time)

    return np.array(arrival_times), np.array(start_times), np.array(finish_times)

def mmc_queue(arrival_rate, service_rate, servicers, cutoff = np.inf, total = False):
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

    mean_queue_length = (
        probability_empty * rho * (servicers * rho) ** servicers /
        ((1 - rho) ** 2 * factorial(servicers))
        )

    if total:

        waiting_time = mean_queue_length ** 2 / arrival_rate

    else:

        waiting_time = mean_queue_length / arrival_rate

    waiting_time[rho == 0] = 0
    waiting_time[rho >= 1] = cutoff
    waiting_time[np.isnan(rho)] = cutoff

    return waiting_time

def probability_available(arrival_rate, service_rate, servicers):
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

    # n = servicers - 1

    probability = 0

    for n in range(servicers):

        probability += (n * rho) ** n / factorial(n) * probability_empty

    return probability_empty

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
        self.c = kwargs.get('c', np.arange(0, 101))
        self.cutoff = kwargs.get('cutoff', np.inf)
        self.bounds = kwargs.get('bounds', (0, np.inf))
        self.initial_guess = kwargs.get('initial_guess', [1.25, 1, .15])
        self.total = kwargs.get('total', False)
        self.p = kwargs.get('p', 1)

        self.build()

    def interpolate(self, rho, c):

        c = np.atleast_2d(c)

        c[c > max(self.c)] = max(self.c)

        result = np.clip(self.interpolator((c, rho)), *self.bounds)

        result[np.isnan(result)] = 0
        result[np.isinf(result)] = 0

        return result

    def available(self, rho, c):

        c = np.atleast_2d(c)

        c[c > max(self.c)] = max(self.c)

        result = np.clip(self.available_interpolator((c, rho)), *self.bounds)

        result[np.isnan(result)] = 0
        result[np.isinf(result)] = 0

        return result

    def max_rho(self, w):

        rho = np.array(
            [np.interp(w, self.waiting_times[idx], self.rho) \
            for idx, c in enumerate(self.c)]
            )

        return rho

    def build(self):

        waiting_times = np.zeros((len(self.c), len(self.rho)))
        self.waiting_times = np.zeros((len(self.c), len(self.rho)))
        # self.probability_available = np.zeros((len(self.c), len(self.rho)))

        # self.probability_functional = np.atleast_2d(binom(ci, .5).pmf(c))

        for idx, c in enumerate(self.c):

            arrival_rate = self.rho * c * self.m

            waiting_times[idx] = mmc_queue(
                arrival_rate, self.m, c, total = self.total, cutoff = self.cutoff,
                )

            # print(waiting_times)

            # self.probability_available[idx] = probability_available(
            #     arrival_rate, self.m, c
            #     )

        for idx, ci in enumerate(self.c):

            pc = np.atleast_2d(binom(ci, self.p).pmf(self.c))
            self.waiting_times[idx] = pc @ waiting_times

        self.interpolator = RegularGridInterpolator(
            (self.c, self.rho), self.waiting_times,
            bounds_error = False, fill_value = np.inf,
            )

        # self.available_interpolator = RegularGridInterpolator(
        #     (self.c, self.rho), self.probability_available,
        #     bounds_error = False, fill_value = 0,
        #     )