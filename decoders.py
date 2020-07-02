import numpy as np
import random as rand
import copy
import collections


from numba import jit, njit
from src.toric_model import Toric_code
from src.planar_model import *
from src.util import *
from src.mcmc import *

from src.mcmc import MCMCDataReader
from src.mcmc import Chain
import pandas as pd
import time

from math import log, exp
from operator import itemgetter


def single_temp(init_code, p, max_iters):
    nbr_eq_classes = init_code.nbr_eq_classes
    ground_state = init_code.define_equivalence_class()
    ladder = [] # list of chain objects
    nbr_errors_chain = np.zeros((nbr_eq_classes, max_iters))

    num_points = 100
    freq = int(max_iters/num_points)
    mean_array = np.zeros((nbr_eq_classes, num_points-1))
    counter = 0

    for eq in range(nbr_eq_classes):
        ladder.append(Chain(init_code.system_size, p, copy.deepcopy(init_code)))
        ladder[eq].code.qubit_matrix = ladder[eq].code.to_class(eq) # apply different logical operator to each chain

    for eq in range(nbr_eq_classes):
        for j in range(max_iters):
            ladder[eq].update_chain(5)
            nbr_errors_chain[eq ,j] = ladder[eq].code.count_errors()

            if j > 0 and j%freq == 0:
                mean_array[eq][counter] = np.average(nbr_errors_chain[eq ,:j])
                counter+=1
        counter = 0
    return mean_array.round(decimals=2)

def STDC(init_code, size, p_error, p_sampling, steps=20000):

    # Create chain with p_sampling, this is allowed since N(n) is independet of p.
    chain = Chain(size, p_sampling, copy.deepcopy(init_code))

    # this is either 4 or 16, depending on what type of code is used.
    nbr_eq_classes = init_code.nbr_eq_classes

    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = [{},{},{},{}]


    num_points = 100
    raindrops = int(steps/100)

    freq = int(steps/num_points)

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros((nbr_eq_classes, num_points))


    # error-model
    counter = 0
    beta = -log((p_error / 3) / (1 - p_error))
    chain_list = []
    for eq in range(nbr_eq_classes):
        chain = Chain(size, p_sampling, copy.deepcopy(init_code))
        chain.code.qubit_matrix = init_code.to_class(eq)
        chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()
        chain_list.append(chain)

    total_counts = 0


    for i in range(num_points):
        for eq in range(nbr_eq_classes):
            # go to class eq and apply stabilizers
            for _ in range(int(steps/num_points)):
                total_counts+=1
                if int(total_counts/4)%int(steps/raindrops) == 0:
                    #print("STDC", int(total_counts/4))
                    chain_list[eq].code.qubit_matrix = chain_list[eq].code.apply_stabilizers_uniform()
                chain_list[eq].update_chain(5)
                # add to dict (only gets added if it is new)
                qubitlist[eq][chain.code.qubit_matrix.tostring()] = np.count_nonzero(chain_list[eq].code.qubit_matrix)

            # compute Z_E
            #print(eqdistr[eq, counter],'here')
            for key in qubitlist[eq]:
                eqdistr[eq, counter] += exp(-beta * qubitlist[eq][key])
        counter+=1

    # Retrun normalized eq_distr
    return eqdistr


def STRC(init_code, size, p_error, p_sampling=None, steps=20000):
    nbr_eq_classes = init_code.nbr_eq_classes
    num_points = 100
    raindrops = raindrops = int(steps/100)

    p_sampling = p_sampling or p_error
    beta_error = -log((p_error / 3) / (1 - p_error))
    beta_sampling = -log((p_sampling / 3) / (1 - p_sampling))
    d_beta = beta_sampling - beta_error

    Z_arr = np.zeros((nbr_eq_classes, num_points))
    max_length = 2 * size ** 2


    #chain = Chain(size, p_sampling, copy.deepcopy(init_code))  # this p needs not be the same as p, as it is used to determine how we sample N(n)

    chain_list = []

    for eq in range(nbr_eq_classes):
        chain = Chain(size, p_sampling, copy.deepcopy(init_code))
        chain.code.qubit_matrix = init_code.to_class(eq)
        chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()
        chain_list.append(chain)

    unique_lengths = [{},{},{},{}]
    len_counts = [{},{},{},{}]
    short_stats_list = [[{'n':max_length, 'N':0} for _ in range(2)],
                        [{'n':max_length, 'N':0} for _ in range(2)],
                        [{'n':max_length, 'N':0} for _ in range(2)],
                        [{'n':max_length, 'N':0} for _ in range(2)]]
    counts  = 0
    total_counts = 0

    for i in range(num_points):
        for eq in range(nbr_eq_classes):
            #unique_lengths = {}
            #len_counts = {}
            # List where first (last) element is stats of shortest (next shortest) length
            # n is length of chain. N is number of unique chains of this length
            short_stats = short_stats_list[eq]
            #chain.code = init_code
            # Apply logical operators to get qubit_matrix into equivalence class i
            #chain.code.qubit_matrix = chain.code.to_class(eq)

            for step in range(int(steps/num_points)):
                total_counts+=1
                if int(total_counts/4)%int(steps/raindrops) == 0:
                    #print("STRC", int(total_counts/4))
                    chain_list[eq].code.qubit_matrix = chain_list[eq].code.apply_stabilizers_uniform()

                chain_list[eq].update_chain(5)
                key = chain_list[eq].code.qubit_matrix.tostring()

                # Check if this error chain has already been seen
                if key in unique_lengths[eq]:
                    # Increment counter for chains of this length
                    len_counts[eq][unique_lengths[eq][key]] += 1

                # If this chain is new, add it to dictionary of unique chains
                else:
                    # Calculate length of this chain
                    length = chain_list[eq].code.count_errors()
                    # Store number of observations and length of this chain
                    unique_lengths[eq][key] = length

                    # Check if this length has been seen before
                    if length in len_counts[eq]:
                        len_counts[eq][unique_lengths[eq][key]] += 1

                        # Otherwise, check if this chain is same length as previous shortest chain
                        if length == short_stats[0]['n']:
                            # Then increment counter of unique chains of shortest length
                            short_stats[0]['N'] += 1

                        # Otherwise, check if this chain same length as previous next shortest chain
                        elif length == short_stats[1]['n']:
                            # Then increment counter of unique chains of next shortest length
                            short_stats[1]['N'] += 1

                    else:
                        # Initiate counter for chains of this length
                        len_counts[eq][unique_lengths[eq][key]] = 1
                        # Check if this chain is shorter than prevous shortest chain
                        if length < short_stats[0]['n']:
                            # Then the previous shortest length is the new next shortest
                            short_stats[1] = short_stats[0]
                            # And the current length is the new shortest
                            short_stats[0] = {'n':length, 'N':1}

                        # Otherwise, check if this chain is shorter than previous next shortest chain
                        elif length < short_stats[1]['n']:
                            # Then reset stats of next shortest chain
                            short_stats[1] = {'n':length, 'N':1}

            # Calculate Boltzmann factor for eq from observed chain lengths
            shortest = short_stats[0]['n']
            shortest_count = short_stats[0]['N']
            shortest_fraction = shortest_count / len_counts[eq][shortest]

            next_shortest = short_stats[1]['n']
            next_shortest_count = short_stats[1]['N']

            if next_shortest != max_length:
                next_shortest_fraction = next_shortest_count / len_counts[eq][next_shortest]
                mean_fraction = 0.5 * (shortest_fraction + next_shortest_fraction * exp(-beta_sampling * (next_shortest - shortest)))

            else:
                mean_fraction = shortest_fraction

            Z_e = sum([m * exp(-beta_sampling * shortest + d_beta * l) for l, m in len_counts[eq].items()]) * mean_fraction

            Z_arr[eq, counts] = Z_e

            short_stats_list[eq] = short_stats
        counts+=1
    return Z_arr


if __name__ == '__main__':
    init_code = Planar_code(5)
    p=0.20
    init_code.generate_random_error(p)
    max_iters = 20000
    x = single_temp(init_code, p, max_iters)
    print(x)
