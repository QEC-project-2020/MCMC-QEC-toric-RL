import numpy as np
import random as rand
import matplotlib.pyplot as plt


from numba import jit, njit
from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.util import *
from src.mcmc import *
<<<<<<< HEAD
from src.mcmc import MCMCDataReader
from src.mcmc import Chain
=======
>>>>>>> 63c4070d93115f346a0f5661a33ad59359cfe3dc
import pandas as pd
import time

from math import log, exp
from operator import itemgetter

def single_temp(init_code, p, max_iters, eps, burnin=625, conv_criteria='error_based'):
    nbr_eq_classes = init_code.nbr_eq_classes
    ground_state = init_code.define_equivalence_class()
    ladder = [] # list of chain objects
<<<<<<< HEAD
    nbr_errors_chain = np.zeros((16, max_iters))
    convergence_reached = np.zeros(16)
    mean_array = np.zeros(16)

    eq_array_translate = np.zeros(16)

    for i in range(nbr_eq_class):
        ladder.append(Chain(init_toric.system_size, p))
        ladder[i].toric = copy.deepcopy(init_toric)  # give all the same initial state
        ladder[i].toric.qubit_matrix = apply_logical_operator(ladder[i].toric.qubit_matrix, i) # apply different logical operator to each chain
        eq_array_translate[i] = define_equivalence_class(ladder[i].toric.qubit_matrix)
    print(eq_array_translate)
    for i in range(nbr_eq_class):
        for j in range(max_iters):
            ladder[i].update_chain(1)
            nbr_errors_chain[i ,j] = np.count_nonzero(ladder[i].toric.qubit_matrix)
            if not convergence_reached[i] and j >= burnin:
                if conv_criteria == 'error_based' and j%100 == 0:
                    convergence_reached[i] = conv_crit_error_based(nbr_errors_chain[i, :j], j, eps)
                    if convergence_reached[i] == 1:
                        mean_array[i] = np.average(nbr_errors_chain[i ,:j])
                        print(j, 'convergence iterations')
                        break
            if not convergence_reached[i] and j == max_iters - 1:
                mean_array[i] = np.average(nbr_errors_chain[i ,:j])
    #print(ground_state, 'ground state')
    return mean_array, convergence_reached, eq_array_translate
=======
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
>>>>>>> 63c4070d93115f346a0f5661a33ad59359cfe3dc


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


<<<<<<< HEAD
def apply_logical_operator(qubit_matrix, number):
    binary = "{0:4b}".format(number)
    for i in range(16):

        if binary[0] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=0, X_pos=0, Z_pos=0)
        if binary[1] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=0, X_pos=0, Z_pos=0)
        if binary[2] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=1, X_pos=0, Z_pos=0)
        if binary[3] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=1, X_pos=0, Z_pos=0)

        return qubit_matrix


=======
>>>>>>> 63c4070d93115f346a0f5661a33ad59359cfe3dc
# add eq-crit that runs until a certain number of classes are found or not?
# separate eq-classes? qubitlist for diffrent eqs
# vill göra detta men med mwpm? verkar finnas sätt att hitta "alla" kortaste, frågan är om man även kan hitta alla längre också
# https://stackoverflow.com/questions/58605904/finding-all-paths-in-weighted-graph-from-node-a-to-b-with-weight-of-k-or-lower
# i nuläget kommer "bra eq" att bli straffade eftersom att de inte kommer få chans att generera lika många unika kedjor --bör man sätta något tak? eller bara ta med de kortaste inom varje?


<<<<<<< HEAD
def single_temp_direct_sum(qubit_matrix, size, p, steps=20000, convcrit = 'error_based'):
    converged = False
    init_toric = Toric_code(size)
    init_toric.qubit_matrix = qubit_matrix
    chain = Chain(size, p)  # this p needs not be the same as p, as it is used to determine how we sample N(n)
=======
def STDC(init_code, size, p_error, p_sampling, steps=20000):
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
    nbr_eq_classes = init_code.nbr_eq_classes

    p_sampling = p_sampling or p_error
    beta_error = -log((p_error / 3) / (1 - p_error))
    beta_sampling = -log((p_sampling / 3) / (1 - p_sampling))
    d_beta = beta_sampling - beta_error

    Z_arr = np.zeros(nbr_eq_classes)
    max_length = 2 * size ** 2

    samples = int(0.9 * steps)
>>>>>>> 63c4070d93115f346a0f5661a33ad59359cfe3dc

    chain = Chain(size, p_sampling, copy.deepcopy(init_code))  # this p needs not be the same as p, as it is used to determine how we sample N(n)

<<<<<<< HEAD
    nbr_errors_chain = np.zeros((16, steps))

    for i in range(16):
        chain.toric.qubit_matrix = apply_logical_operator(qubit_matrix, i)  # apply different logical operator to each chain
        # We start in a state with high entropy, therefore we let mcmc "settle down" before getting samples.
        current_class = define_equivalence_class(chain.toric.qubit_matrix)
        for _ in range(int(steps*0.8)):
=======
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
>>>>>>> 63c4070d93115f346a0f5661a33ad59359cfe3dc
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

        # Calculate Boltzmann factor for eq from observed chain lengths
        shortest = short_stats[0]['n']
        shortest_count = short_stats[0]['N']
        shortest_fraction = shortest_count / len_counts[shortest]

        next_shortest = short_stats[1]['n']
        next_shortest_count = short_stats[1]['N']
        
        if next_shortest != max_length:
            next_shortest_fraction = next_shortest_count / len_counts[next_shortest]
            mean_fraction = 0.5 * (shortest_fraction + next_shortest_fraction * exp(-beta_sampling * (next_shortest - shortest)))
        
        else:
            mean_fraction = shortest_fraction

        Z_e = sum([m * exp(-beta_sampling * shortest + d_beta * l) for l, m in len_counts.items()]) * mean_fraction

        Z_arr[eq] = Z_e

    return (Z_arr / np.sum(Z_arr) * 100).astype(dtype=int)


<<<<<<< HEAD
    return (np.divide(eqdistr, sum(eqdistr)) * 100).astype(np.uint8)






if __name__ == '__main__':
    init_toric = Toric_code(5)
    p_error = 0.19
    #init_toric.generate_random_error(p_error)
    data_reader = MCMCDataReader('training_data/data_5x5_p_0.19', init_toric.system_size)
    success = 0
    iters = 100
    for j in range(iters):
        init_toric.qubit_matrix, distr = data_reader.next()
        result = single_temp_direct_sum(init_toric.qubit_matrix, init_toric.system_size, p = p_error, steps=10000)
        print(result, 'STDS')
        print(distr, 'MCMC')
        if np.argmax(result) == np.argmax(distr): success+=1
        print(j)
    print('same as MCMC for this many syndromes: ', success/iters)


    """mean_array, convergence_reached, eq_array_translate = single_temp(init_toric, p = p_error, max_iters = 1000000, eps = 0.00001, burnin = 100000, conv_criteria = 'error_based')
    print(eq_array_translate[np.argmin(mean_array)], 'guess')
    print(convergence_reached)"""
    '''init_toric = Toric_code(5)
    p_error = 0.15
    init_toric.generate_random_error(p_error)
    print(init_toric.qubit_matrix)
    print(single_temp_direct_sum(init_toric.qubit_matrix, size=5, p=p_error, steps=20000))'''
=======
if __name__ == '__main__':
    t0 = time.time()
    size =  7
    steps = 10000 * int(1 + (size / 5) ** 4)
    #reader = MCMCDataReader('data/data_7x7_p_0.19.xz', size)
    p_error = 0.2
    p_sampling = 0.2
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

            v1, most_likeley_eq, convergece = single_temp(init_code, p=p_error, max_iters=steps, eps=0.005, conv_criteria = None)
            print('Try single_temp', i+1, ':', v1, 'most_likely_eq', most_likeley_eq, 'ground state:', ground_state, 'convergence:', convergece, time.time()-t0)
            t0 = time.time()
            distrs[i] = STDC(copy.deepcopy(init_code), size=size, p=p_error, steps=steps)
            #distrs[i] = STRC(copy.deepcopy(init_code), size=size, p_error=p_error, p_sampling=p_sampling, steps=steps)
            print('Try STDC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'ground state:', ground_state, time.time()-t0)
            t0 = time.time()

        tvd = sum(abs(distrs[1]-distrs[0]))
        mean_tvd += tvd
        print('TVD:', tvd)
    print('Mean TVD:', mean_tvd / 10)

        #print('STRC distribution 1:', distr1)
        #print('STRC distribution 2:', distr2)
>>>>>>> 63c4070d93115f346a0f5661a33ad59359cfe3dc
