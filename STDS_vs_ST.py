import numpy as np
import random as rand
import collections
import time
from numba import jit, prange
from src.toric_model import Toric_code
from src.util import Action
from src.mcmc import *
from src.mcmc import Chain
import pandas as pd
from copy import deepcopy
from src.toric_model import Toric_code
from src.mcmc import Chain, define_equivalence_class, apply_logical
from math import log, exp

from single_temperature import single_temp
from single_temperature_direct_sum import single_temp_direct_sum

def main():
    t0 = time.time()
    size = 7
    init_toric = Toric_code(size)
    p_error = 0.15
    init_toric.generate_random_error(p_error)
    print(init_toric.qubit_matrix)
    print(single_temp_direct_sum(init_toric.qubit_matrix, size=size, p=p_error, steps=30000), "STDS result")
    print(time.time()-t0, 'STDS time')
    mean_array, convergence_reached, eq_array_translate = single_temp(init_toric, p = p_error, max_iters = 10000000, eps = 0.0001, burnin = 50000, conv_criteria = 'error_based')
    print(eq_array_translate[np.argmin(mean_array)], 'ST most likeley eq class')
    print(convergence_reached)
    print(time.time()-t0, 'ST time')

if __name__ == '__main__':
    main()
