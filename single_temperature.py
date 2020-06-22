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

def apply_logical_operator(qubit_matrix, number):
    binary = "{0:4b}".format(number)
    for i in range(16):

        if binary[0] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=0, X_pos=0, Z_pos=0)
        if binary[1] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=0, X_pos=0, Z_pos=0)
        if binary[2] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=1, X_pos=0, Z_pos=0)
        if binary[3] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=1, X_pos=0, Z_pos=0)

        return qubit_matrix

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
def main1():
    data_reader = MCMCDataReader('./training_data/data_5x5_p_0.19', 1)
    init_toric = Toric_code(5)
    p_error = np.linspace(0.1, 0.75, 50)
    n = np.zeros((16, len(p_error)))

    #init_toric.qubit_matrix, distr = data_reader.next()

    #init_toric.generate_random_error(0.19)
    #init_toric.qubit_matrix =np.array([[[0, 1, 0, 0, 0],[0, 0, 0, 0, 0],[0, 1, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 1, 2, 0]],[[0, 0, 2, 0, 0],[0, 0, 0, 0, 0],[1, 1, 0, 2, 0],[0, 0, 0, 0, 0],[0, 1, 0, 2, 0]]])
    #init_toric.qubit_matrix =np.array([[[0, 0, 0, 2, 1],[0, 0, 0, 2, 0],[0, 0 ,0, 0, 0],[0, 0, 0, 0, 0,],[2, 2, 0, 0, 0,]],[[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 2, 3, 0, 0],[0, 0, 1, 2, 0],[0, 0, 0, 2, 0]]])
    init_toric.qubit_matrix = np.array([[[0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0],
                                         [0, 1, 2, 1, 1, 0, 0],
                                         [0, 0, 3, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0]],
                                        [[0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0],
                                         [0, 0, 2, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0],
                                         [0, 3, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0]]])
    ground_state = define_equivalence_class(init_toric.qubit_matrix)
    print(init_toric.qubit_matrix)
    #init_toric.qubit_matrix, _ = apply_random_logical(init_toric.qubit_matrix)
    #init_toric.qubit_matrix = apply_stabilizers_uniform(init_toric.qubit_matrix)
    for i in range(len(p_error)):
        #mean_array, convergence_reached, eq_array_translate = single_temp(init_toric, p = p_error[i], max_iters = 1000000, eps = 0.00001, burnin =50000, conv_criteria = 'error_based')
        #mean_array, convergence_reached, eq_array_translate = single_temp(init_toric, p = p_error[i], max_iters = 1000, eps = 0.1, burnin =1, conv_criteria = 'error_based')
        mean_array, convergence_reached, eq_array_translate = single_temp(init_toric, p = p_error[i], max_iters = 1000000, eps = 0.0001, burnin =50000, conv_criteria = 'error_based')
        print(eq_array_translate[np.argmin(mean_array)], 'single temp')
        print(ground_state, 'ground_state')
        #print(np.argmax(distr), 'mcmc result')
        print(convergence_reached, 'convergence')
        n[:,i] = mean_array
    for j in range(16):
        plt.plot(-np.log(p_error/3/(1-p_error)), n[j,:])
    plt.show()

def main2():
    size = 5
    data_reader = MCMCDataReader('./training_data/data_5x5_p_0.16', size)
    init_toric = Toric_code(size)
    p_error = 0.16
    corresponding = 0
    success_rate = 0
    iterations = 10
    for j in range(iterations):
        init_toric.qubit_matrix, distr = data_reader.next()
        ground_state = define_equivalence_class(init_toric.qubit_matrix)
        print(init_toric.qubit_matrix)
        #init_toric.qubit_matrix, _ = apply_random_logical(init_toric.qubit_matrix)
        #init_toric.qubit_matrix = apply_stabilizers_uniform(init_toric.qubit_matrix)
        mean_array, convergence_reached, eq_array_translate = single_temp(init_toric, p = p_error, max_iters = 2000000, eps = 0.001, burnin = 50000, conv_criteria = 'error_based')
        print(eq_array_translate[np.argmin(mean_array)], 'single temp')
        print(ground_state, 'ground_state')
        print(np.argmax(distr), 'mcmc result')
        print(convergence_reached, 'convergence_reached')
        if np.argmax(distr) == eq_array_translate[np.argmin(mean_array)]: corresponding+=1
        if ground_state == eq_array_translate[np.argmin(mean_array)]: success_rate+=1
    print(corresponding/iterations, 'corresponding/iterations')
    print(success_rate/iterations, 'success_rate/iterations')
if __name__ == '__main__':
    main1()
