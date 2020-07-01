import numpy as np
import random as rand
import pandas as pd

import copy
import collections
import os

from numba import jit, prange
from src.toric_model import Toric_code
from src.util import *
from NN import NN_11, NN_17, NN_11_mod
from tqdm import tqdm
from src.mcmc import *
from src.RL import *


def main(file_path, RL_args, prediction_args):
    size = 5

    names = ['qubit_matrix', 'drl_correction_chain']
    data = []

    rl = RL(**RL_args)
    rl.load_network(prediction_args['PATH'])
    reader = rl.mcmc_data_reader

    for i in tqdm(range(reader.get_capacity())):
        tmp_dict = {}
        qubit_matrix, eq_distr = reader.next()

        success, drl_correction_chain = rl.prediction_mod(**prediction_args, qubit_matrix=np.copy(qubit_matrix)) # Anpassa efter Alexeis kod

        drl_smart = (define_equivalence_class(drl_correction_chain) == np.argmax(eq_distr))
        if success and not drl_smart:
            tmp_dict['qubit_matrix'] = qubit_matrix
            tmp_dict['drl_correction_chain'] = drl_correction_chain
            data.append(tmp_dict)
        
    df = pd.DataFrame(data)
    p_str = file_path[file_path.find('p_0'):file_path.find('.xz')]
    df.to_pickle('data/drl_failures_{}.xz'.format(p_str))


def getMCMCstats(qubit_matrix_in, size, p_error, Nc=19, steps=1000000, crit='error_based'):
    """Get statistics about distribution of error chains given a syndrom (error chain) using MCMC-sampling."""
    init_toric = Toric_code(size)
    
    # define error
    init_toric.qubit_matrix = qubit_matrix_in
    init_toric.syndrom('next_state')

    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')
    # Start in random eq-class
    init_toric.qubit_matrix, _ = apply_random_logical(init_toric.qubit_matrix)

    distr, count, qubitlist = parallel_tempering_plus(init_toric, Nc, p=p_error, steps=steps, iters=10, conv_criteria=crit)
    print(distr)
    unique_elements, unique_counts = np.unique(qubitlist, axis=0, return_counts=True)
    print('Number of unique elements: ', len(unique_elements))

    shortest = 1000
    for i in range(len(qubitlist)):
        nb = np.count_nonzero(qubitlist[i])
        if nb < shortest:
            shortest = nb

    # save all qubits to df_all
    df = pd.DataFrame({"qubit":[], "nbr_err":[], "nbr_occ":[], "eq_class":[]})
    df = pd.concat((pd.DataFrame({"qubit":[unique_elements[i]], "nbr_err":[np.count_nonzero(unique_elements[i])], "nbr_occ":[unique_counts[i]], "eq_class": define_equivalence_class(unique_elements[i])}) for i in range(len(unique_elements))),
            ignore_index=True)
    
    for i in range(7):
        print(shortest+i)
        print(df.loc[df['nbr_err'] == shortest + i])
        for j in range(16):
            nbr_comb = len(df.loc[df['nbr_err'] == shortest + i].loc[df['eq_class'] == j])
            if nbr_comb > 0:
                print('class ', j, '\t\t', nbr_comb)


if __name__ == '__main__':
    size = 5
    p_error = 0.15
    file_path = './data/data_5x5_p_0.15.xz'
    prediction_args = {'prediction_list_p_error': [0.0], 'PATH': './network/Size_5_NN_11.pt'}
    RL_args = {'Network': NN_11, 'Network_name': 'Size_5_NN_11', 'system_size': size, 
        'p_error': p_error, 'replay_memory_capacity': 1e4, 'DATA_FILE_PATH': file_path}
    main(file_path, RL_args, prediction_args)