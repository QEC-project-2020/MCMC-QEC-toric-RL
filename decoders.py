import numpy as np
import random as rand
import copy
import collections


from numba import jit, njit
from src.toric_model import Toric_code
from src.planar_model import Planar_code
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
