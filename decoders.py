import numpy as np
import random as rand
import matplotlib.pyplot as plt


from numba import jit, prange
from src.toric_model import Toric_code
from src.util import Action
from src.mcmc import *
from src.mcmc import MCMCDataReader
from src.mcmc import Chain
import pandas as pd

from math import log, exp

def single_temp(init_toric, p, max_iters, eps, burnin = 625, conv_criteria = 'error_based'):
    nbr_eq_class = 16
    ground_state = define_equivalence_class(init_toric.qubit_matrix)
    ladder = [] # list of chain objects
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


def apply_logical_operator(qubit_matrix, number):
    binary = "{0:4b}".format(number)
    for i in range(16):

        if binary[0] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=0, X_pos=0, Z_pos=0)
        if binary[1] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=0, X_pos=0, Z_pos=0)
        if binary[2] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=1, X_pos=0, Z_pos=0)
        if binary[3] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=1, X_pos=0, Z_pos=0)

        return qubit_matrix


# add eq-crit that runs until a certain number of classes are found or not?
# separate eq-classes? qubitlist for diffrent eqs
# vill göra detta men med mwpm? verkar finnas sätt att hitta "alla" kortaste, frågan är om man även kan hitta alla längre också
# https://stackoverflow.com/questions/58605904/finding-all-paths-in-weighted-graph-from-node-a-to-b-with-weight-of-k-or-lower
# i nuläget kommer "bra eq" att bli straffade eftersom att de inte kommer få chans att generera lika många unika kedjor --bör man sätta något tak? eller bara ta med de kortaste inom varje?


def single_temp_direct_sum(qubit_matrix, size, p, steps=20000, convcrit = 'error_based'):
    converged = False
    init_toric = Toric_code(size)
    init_toric.qubit_matrix = qubit_matrix
    chain = Chain(size, p)  # this p needs not be the same as p, as it is used to determine how we sample N(n)

    qubitlist = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]

    nbr_errors_chain = np.zeros((16, steps))

    for i in range(16):
        chain.toric.qubit_matrix = apply_logical_operator(qubit_matrix, i)  # apply different logical operator to each chain
        # We start in a state with high entropy, therefore we let mcmc "settle down" before getting samples.
        current_class = define_equivalence_class(chain.toric.qubit_matrix)
        for _ in range(int(steps*0.8)):
            chain.update_chain(5)
        for _ in range(int(steps*0.2)):
            chain.update_chain(5)
            qubitlist[current_class][chain.toric.qubit_matrix.tostring()] = np.count_nonzero(chain.toric.qubit_matrix)

    # --------Determine EC-Distrubution--------
    eqdistr = np.zeros(16)
    beta = -log((p / 3) / (1-p))

    for i in range(16):
        for key in qubitlist[i]:
            eqdistr[i] += exp(-beta*qubitlist[i][key])

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
