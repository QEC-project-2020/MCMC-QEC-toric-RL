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

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")


def plot_p():
    size = 7
    init_toric = Toric_code(size)
    Nc = 19
    steps=10000000
    
    # define error
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
    init_toric.syndrom('next_state')


    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')
    # Start in random eq-class
    init_toric.qubit_matrix, _ = apply_random_logical(init_toric.qubit_matrix)

    p = (0.01 + 0.03*i for i in range(7))
   
    df = pd.DataFrame()
    for p_err in p:
        distr, count, qubitlist = parallel_tempering_plus(init_toric, Nc, p=p_err, steps=steps, iters=10, conv_criteria='none')
        for i in range(16):
            df = df.append(pd.DataFrame({"p":[p_err], "P":[distr[i]], "eq_class":'kl:'+str(i)}), ignore_index=True)

    #df.columns = ['p', 'P', 'eq_class']

    ax = sns.lineplot(x='p', y='P', hue='eq_class', data=df)
    ax.set_xlabel("Felsannolikhet, $p$")
    ax.set_ylabel("Sannolikhet, $P_eq$")
    #plt.legend(loc='upper right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=5.)
    plt.savefig('plots/testing.png')



if __name__ == '__main__':
    plot_p()
