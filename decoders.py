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

def single_temp(init_code, p, max_iters, eps, burnin=625, conv_criteria='error_based'):
    nbr_eq_classes = init_code.nbr_eq_classes
    ground_state = init_code.define_equivalence_class()
    ladder = [] # list of chain objects
    nbr_errors_chain = np.zeros((nbr_eq_classes, max_iters))
    convergence_reached = np.zeros(nbr_eq_classes)
    mean_array = np.zeros(nbr_eq_classes, dtype=float)

    for eq in range(nbr_eq_classes):
        ladder.append(Chain(init_code.system_size, p, copy.deepcopy(init_code)))
        ladder[eq].code.qubit_matrix = ladder[eq].code.to_class(eq) # apply different logical operator to each chain

    for eq in range(nbr_eq_classes):
        for j in range(max_iters):
            ladder[eq].update_chain(5)
            nbr_errors_chain[eq ,j] = ladder[eq].code.count_errors()
            if not convergence_reached[eq] and j >= burnin:
                if conv_criteria == 'error_based' and not j % 100:
                    convergence_reached[eq] = conv_crit_error_based(nbr_errors_chain[eq, :j], j, eps)
                    if convergence_reached[eq] == 1:
                        mean_array[eq] = np.average(nbr_errors_chain[eq ,:j])
                        break
                #elif conv_criteria == None:
                    #mean_array[eq] = np.average(nbr_errors_chain[eq ,:j])
                    #continue
                    #mean_array[eq] = np.average(nbr_errors_chain[eq ,:j])
            if conv_criteria != None and j == max_iters-1:
                mean_array[eq] = 2*init_code.system_size**2 #not chosen if not converged
            elif j == max_iters-1:
                mean_array[eq] = np.average(nbr_errors_chain[eq ,:j])
    most_likeley_eq = np.argmin(mean_array)
    return mean_array.round(decimals=2), most_likeley_eq, convergence_reached


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


def STDC(init_code, size, p_error, p_sampling=None, steps=20000):
    # set p_sampling equal to p_error by default
    p_sampling = p_sampling or p_error

    # Create chain with p_sampling, this is allowed since N(n) is independet of p.
    chain = Chain(size, p_sampling, copy.deepcopy(init_code))

    # this is either 4 or 16, depending on what type of code is used.
    nbr_eq_classes = init_code.nbr_eq_classes

    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = {}

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)

    # error-model
    beta = -log((p_error / 3) / (1 - p_error))

    for eq in range(nbr_eq_classes):
        # go to class eq and apply stabilizers
        chain.code.qubit_matrix = init_code.to_class(eq)
        chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()

        for _ in range(steps):
            chain.update_chain(5)
            # add to dict (only gets added if it is new)
            qubitlist[chain.code.qubit_matrix.tostring()] = np.count_nonzero(chain.code.qubit_matrix)

        # compute Z_E        
        for key in qubitlist:
            eqdistr[eq] += exp(-beta * qubitlist[key])
        qubitlist.clear()

    # Retrun normalized eq_distr
    return (np.divide(eqdistr, sum(eqdistr)) * 100).astype(np.uint8)


def STRC(init_code, size, p_error, p_sampling=None, steps=20000):
    # set p_sampling equal to p_error by default
    p_sampling = p_sampling or p_error

    # Create chain with p_sampling, this is allowed since N(n) is independet of p.
    chain = Chain(size, p_sampling, copy.deepcopy(init_code))

    # Either 4 or 16, depending on type of code
    nbr_eq_classes = init_code.nbr_eq_classes

    # error model
    beta_error = -log((p_error / 3) / (1 - p_error))
    beta_sampling = -log((p_sampling / 3) / (1 - p_sampling))
    d_beta = beta_sampling - beta_error

    # Array to hold the boltzmann factors for every class
    Z_arr = np.zeros(nbr_eq_classes)

    # Largest possible chain length
    max_length = 2 * size ** 2

    # Iterate through equivalence classes
    for eq in range(nbr_eq_classes):
        unique_lengths = {}
        len_counts = {}
        # List where first (last) element is stats of shortest (next shortest) length
        # n is length of chain. N is number of unique chains of this length
        short_stats = [{'n':max_length, 'N':0} for _ in range(2)]
        chain.code = init_code
        # Apply logical operators to get qubit_matrix into equivalence class i
        chain.code.qubit_matrix = chain.code.to_class(eq)

        # Generate chains
        for step in range(steps):
            # Do metropolis sampling
            chain.update_chain(5)

            # Convert the current qubit matrix to string for hashing
            key = chain.code.qubit_matrix.tostring()

            # Check if this error chain has already been seen by comparing hashes
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

        # Partial result needed for boltzmann factor
        shortest = short_stats[0]['n']
        shortest_count = short_stats[0]['N']
        shortest_fraction = shortest_count / len_counts[shortest]

        next_shortest = short_stats[1]['n']
        next_shortest_count = short_stats[1]['N']
        
        # Handle rare cases where only one chain length is observed
        if next_shortest != max_length:
            next_shortest_fraction = next_shortest_count / len_counts[next_shortest]
            mean_fraction = 0.5 * (shortest_fraction + next_shortest_fraction * exp(-beta_sampling * (next_shortest - shortest)))
        
        else:
            mean_fraction = shortest_fraction

        # Calculate boltzmann factor from observed chain lengths
        Z_e = sum([m * exp(-beta_sampling * shortest + d_beta * l) for l, m in len_counts.items()]) * mean_fraction
        Z_arr[eq] = Z_e

    # Use boltzmann factors as relative probabilities and normalize distribution
    return (Z_arr / np.sum(Z_arr) * 100).astype(dtype=int)


if __name__ == '__main__':
    t0 = time.time()
    size =  7
    steps = 10000 * int(1 + (size / 5) ** 4)
    #reader = MCMCDataReader('data/data_7x7_p_0.19.xz', size)
    p_error = 0.15
    p_sampling = 0.15
    init_code = Planar_code(size)
    tries = 2
    distrs = np.zeros((tries, init_code.nbr_eq_classes))
    mean_tvd = 0.0
    for i in range(10):
        init_code.generate_random_error(p_error)
        ground_state = init_code.define_equivalence_class()
        init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
        init_qubit = np.copy(init_code.qubit_matrix)

        print('################ Chain', i+1 , '###################')

        for i in range(tries):
            #print('Try', i+1, ':', distrs[i], 'most_likeley_eq', np.argmax(distrs[i]), 'ground state:', ground_state)

            v1, most_likely_eq, convergece = single_temp(init_code, p=p_error, max_iters=steps, eps=0.005, conv_criteria = None)
            print('Try single_temp', i+1, ':', v1, 'most_likely_eq', most_likely_eq, 'ground state:', ground_state, 'convergence:', convergece, time.time()-t0)
            t0 = time.time()
            distrs[i] = STDC(copy.deepcopy(init_code), size=size, p_error=p_error, steps=steps)
            #distrs[i] = STRC(copy.deepcopy(init_code), size=size, p_error=p_error, p_sampling=p_sampling, steps=steps)
            print('Try STDC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'ground state:', ground_state, time.time()-t0)
            t0 = time.time()

        tvd = sum(abs(distrs[1]-distrs[0]))
        mean_tvd += tvd
        print('TVD:', tvd)
    print('Mean TVD:', mean_tvd / 10)

        #print('STRC distribution 1:', distr1)
        #print('STRC distribution 2:', distr2)
