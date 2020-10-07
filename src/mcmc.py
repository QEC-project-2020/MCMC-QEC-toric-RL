import numpy as np
import random as rand
import copy

from numba import jit, njit
from .planar_model import _apply_random_stabilizer  # ???
import pandas as pd


class Chain:
    def __init__(self, p, code):
        self.code = code
        self.p = p
        self.p_logical = 0
        self.flag = 0
        self.factor = ((self.p / 3.0) / (1.0 - self.p))  # rename me
    # runs iters number of steps of the metroplois-hastings algorithm

    def update_chain(self, iters):
        if self.p_logical != 0:
            for _ in range(iters):
                # apply logical or stabilizer with p_logical
                if rand.random() < self.p_logical:
                    new_matrix, qubit_errors_change = self.code.apply_random_logical()
                else:
                    new_matrix, qubit_errors_change = self.code.apply_random_stabilizer()

                # Avoid calculating r if possible. If self.p is 0.75 r = 1 and we accept all changes
                # If the new qubit matrix has equal or fewer errors, r >= 1 and we also accept all changes
                if self.p >= 0.75 or qubit_errors_change <= 0:
                    self.code.qubit_matrix = new_matrix
                    continue
                # acceptence ratio
                if rand.random() < self.factor ** qubit_errors_change:
                    self.code.qubit_matrix = new_matrix

        else:
            for _ in range(iters):
                new_matrix, qubit_errors_change = self.code.apply_random_stabilizer()

                # acceptence ratio
                if rand.random() < self.factor ** qubit_errors_change:
                    self.code.qubit_matrix = new_matrix

    def update_chain_fast(self, iters):
        self.code.qubit_matrix = _update_chain_fast(self.code.qubit_matrix, self.factor, iters)


class Ladder:
    def __init__(self, p_bottom, init_code, Nc, p_logical=0):
        # sampling probability of bottom chain
        self.p_bottom = p_bottom

        # seed code
        self.init_code = init_code

        # number of chains
        self.Nc = Nc

        # logical sampling rate in top chain
        self.p_logical = p_logical
        p_top = 0.75

        # temporary list of sampling probabilities
        p_ladder = np.linspace(p_bottom, p_top, Nc)
        self.p_ladder = p_ladder

        # list of relative probabilities
        self.p_diff = (p_ladder[:-1] * (1 - p_ladder[1:])) / (p_ladder[1:] * (1 - p_ladder[:-1]))

        # list of Chains of increasing p
        self.chains = [Chain(p, copy.deepcopy(init_code)) for p in p_ladder]

        # special properties of top chain
        self.chains[-1].flag = 1
        self.chains[-1].p_logical = p_logical

        # count of chains that have "fallen all the way down"
        self.tops0 = 0

    def update_ladder(self, iters):
        for chain in self.chains:
            chain.update_chain(iters)

    # returns true if flip should be performed
    def r_flip(self, ind_lo):
        # chain lengths
        ne_lo = self.chains[ind_lo].code.count_errors()
        ne_hi = self.chains[ind_lo + 1].code.count_errors()
        # relative probabilities between chains (except exponent)
        rel_p = self.p_diff[ind_lo]
        return _r_flip(ne_lo, ne_hi, rel_p)

    def step(self, iters):
        self.update_ladder(iters)
        for i in reversed(range(self.Nc - 1)):
            if self.r_flip(i):
                self.chains[i].code, self.chains[i + 1].code = self.chains[i + 1].code, self.chains[i].code
                self.chains[i].flag, self.chains[i + 1].flag = self.chains[i + 1].flag, self.chains[i].flag
        self.chains[-1].flag = 1
        if self.chains[0].flag == 1:
            self.tops0 += 1
            self.chains[0].flag = 0


# This is the object we crate to read a file during training
class MCMCDataReader:
    def __init__(self, file_path, size):
        # file_path needs to be dataframe in pickle format
        self.__file_path = file_path
        # size is the size of the toric code
        self.__size = size
        try:
            self.__df = pd.read_pickle(file_path)
            self.__capacity = self.__df.index[-1][0] + 1  # The number of data samples in the dataset
        except:  # TODO fix exception
            print('No input file for MCMCDataReader')
        self.__current_index = 0

    def full(self):
        return self.__df.to_numpy().ravel()

    def has_next(self):
        return self.__current_index < self.__capacity

    def current_index(self):
        return self.__current_index

    def get_capacity(self):
        return self.__capacity


@njit('(int64, int64, float64)')
def _r_flip(ne_lo, ne_hi, rel_p):
    if ne_hi < ne_lo:
        return True
    else:
        return rand.random() < rel_p ** (ne_hi - ne_lo)


@njit(cache=True)
def _update_chain_fast(qubit_matrix, factor, iters):
    for _ in range(iters):
        new_matrix, qubit_errors_change = _apply_random_stabilizer(qubit_matrix)

        # acceptence ratio
        if rand.random() < factor ** qubit_errors_change:
            qubit_matrix = new_matrix
    return qubit_matrix
