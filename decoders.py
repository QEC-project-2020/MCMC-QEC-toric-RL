import numpy as np
import random as rand
import copy
import collections


from numba import jit, njit
from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.util import *
from src.mcmc import *
import pandas as pd
import time

from math import log, exp
from operator import itemgetter

def single_temp(init_code, p, max_iters):
    nbr_eq_classes = init_code.nbr_eq_classes
    ground_state = init_code.define_equivalence_class()
    ladder = [] # list of chain objects
    nbr_errors_chain = np.zeros((nbr_eq_classes, max_iters))
    mean_array = np.zeros((nbr_eq_classes, max_iters))

    for eq in range(nbr_eq_classes):
        ladder.append(Chain(init_code.system_size, p, copy.deepcopy(init_code)))
        ladder[eq].code.qubit_matrix = ladder[eq].code.to_class(eq) # apply different logical operator to each chain

    for eq in range(nbr_eq_classes):
        for j in range(max_iters):
            ladder[eq].update_chain(5)
            nbr_errors_chain[eq ,j] = ladder[eq].code.count_errors()
            mean_array[eq][j] = np.average(nbr_errors_chain[eq ,:j])
    return mean_array.round(decimals=2)


def conv_crit_error_based(nbr_errors_chain, l, eps):  # Konvergenskriterium 1 i papper
    # last nonzero element of nbr_errors_bottom_chain is since_burn. Length of nonzero part is since_burn + 1
    # Calculate average number of errors in 2nd and 4th quarter
    Average_Q2 = np.average(nbr_errors_chain[(l // 4): (l // 2)])
    Average_Q4 = np.average(nbr_errors_chain[(3 * l // 4): l])

    # Compare averages
    error = abs(Average_Q2 - Average_Q4)

    if error < eps:
        return 1
    else:
        return 0


# add eq-crit that runs until a certain number of classes are found or not?
# separate eq-classes? qubitlist for diffrent eqs
# vill göra detta men med mwpm? verkar finnas sätt att hitta "alla" kortaste, frågan är om man även kan hitta alla längre också
# https://stackoverflow.com/questions/58605904/finding-all-paths-in-weighted-graph-from-node-a-to-b-with-weight-of-k-or-lower
# i nuläget kommer "bra eq" att bli straffade eftersom att de inte kommer få chans att generera lika många unika kedjor --bör man sätta något tak? eller bara ta med de kortaste inom varje?


def single_temp_direct_sum(init_code, size, p, steps=20000):
    chain = Chain(size, p, copy.deepcopy(init_code))  # this p needs not be the same as p, as it is used to determine how we sample N(n)
    nbr_eq_classes = init_code.nbr_eq_classes

    qubitlist = [{} for _ in range(nbr_eq_classes)]

    for eq in range(nbr_eq_classes):
        #chain.code.qubit_matrix = apply_logical_operator(qubit_matrix, i)  # apply different logical operator to each chain
        chain.code.qubit_matrix = init_code.to_class(eq)
        # We start in a state with high entropy, therefore we let mcmc "settle down" before getting samples.
        for _ in range(int(steps*0.8)):
            chain.update_chain(5)
        for _ in range(int(steps*0.2)):
            chain.update_chain(5)
            qubitlist[eq][chain.code.qubit_matrix.tostring()] = chain.code.count_errors()

    # --------Determine EC-Distrubution--------
    eqdistr = np.zeros(nbr_eq_classes)
    beta = -log((p / 3) / (1-p))

    for eq in range(nbr_eq_classes):
        for key in qubitlist[eq]:
            eqdistr[eq] += exp(-beta * qubitlist[eq][key])

    return (np.divide(eqdistr, sum(eqdistr)) * 100).astype(np.uint8)


def single_temp_relative_count(init_code, size, p_error, p_sampling=None, steps=20000):
    nbr_eq_classes = init_code.nbr_eq_classes

    p_sampling = p_sampling or p_error
    beta_error = -log((p_error / 3) / (1 - p_error))
    beta_sampling = -log((p_sampling / 3) / (1 - p_sampling))
    d_beta = beta_sampling - beta_error

    Z_arr = np.zeros(nbr_eq_classes)
    max_length = 2 * size ** 2

    samples = int(0.9 * steps)

    chain = Chain(size, p_sampling, copy.deepcopy(init_code))  # this p needs not be the same as p, as it is used to determine how we sample N(n)

    for eq in range(nbr_eq_classes):
        unique_lengths = {}
        len_counts = {}
        # List where first (last) element is stats of shortest (next shortest) length
        # n is length of chain. N is number of unique chains of this length
        short_stats = [{'n':max_length, 'N':0} for _ in range(2)]
        chain.code = init_code
        # Apply logical operators to get qubit_matrix into equivalence class i
        chain.code.qubit_matrix = chain.code.to_class(eq)

        for _ in range(steps - samples):
            chain.update_chain(5)
        for step in range(samples):
            chain.update_chain(5)

            key = chain.code.qubit_matrix.tostring()

            # Check if this error chain has already been seen
            if key in unique_lengths:
                # Increment counter for chains of this length
                len_counts[unique_lengths[key]] += 1

            # If this chain is new, add it to dictionary of unique chains
            else:
                # Calculate length of this chain
                length = chain.code.count_errors()
                # Store number of observations and length of this chain
                unique_lengths[key] = length

                # Check if this length has been seen before
                if length in len_counts:
                    len_counts[unique_lengths[key]] += 1

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
                    len_counts[unique_lengths[key]] = 1
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

        # Dict to hold the total occurences and unique chains of each observed length
        shortest = short_stats[0]['n']
        shortest_count = short_stats[0]['N']
        next_shortest = short_stats[1]['n']
        next_shortest_count = short_stats[1]['N']

        shortest_fraction = shortest_count / len_counts[shortest]
        next_shortest_fraction = next_shortest_count / len_counts[next_shortest]
        mean_fraction = 0.5 * (shortest_fraction + next_shortest_fraction * exp(-beta_sampling * (next_shortest - shortest)))

        Z_e = sum([m * exp(-beta_sampling * shortest + d_beta * l) for l, m in len_counts.items()]) * mean_fraction

        Z_arr[eq] = Z_e

    return (Z_arr / np.sum(Z_arr) * 100).astype(dtype=int)


if __name__ == '__main__':
    t0 = time.time()
    size =  10
    steps = 10000 * int(1 + (size / 5) ** 4)#reader = MCMCDataReader('data/data_7x7_p_0.19.xz', size)
    p_error = 0.2
    p_sampling = 0.2
    init_code = Planar_code(size)
    init_code.generate_random_error()
    single_temp(init_code, p=p_error, max_iters =100)
