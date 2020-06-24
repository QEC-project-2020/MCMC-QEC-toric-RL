import numpy as np
import random as rand
import copy
import collections

from numba import jit, prange
from .toric_model import Toric_code
from .util import Action
import pandas as pd

# replace with xor?
rule_table = np.array(([[0, 1, 2, 3], [1, 0, 3, 2],
                        [2, 3, 0, 1], [3, 2, 1, 0]]), dtype=int)    # Identity = 0
                                                                    # pauli_x = 1
                                                                    # pauli_y = 2
                                                                    # pauli_z = 3


class Chain:
    def __init__(self, size, p):
        self.toric = Toric_code(size)
        self.size = size
        self.p = p
        self.p_logical = 0
        self.flag = 0
    # runs iters number of steps of the metroplois-hastings algorithm
    def update_chain(self, iters):
        for _ in range(iters):
            # apply logical or stabilizer with p_logical
            if rand.random() < self.p_logical:
                new_matrix, qubit_errors_change = apply_random_logical(self.toric.qubit_matrix)
            else:
                new_matrix, qubit_errors_change = apply_random_stabilizer(self.toric.qubit_matrix)

            # Avoid calculating r if possible. If self.p is 0.75 r = 1 and we accept all changes
            # If the new qubit matrix has equal or fewer errors, r >= 1 and we also accept all changes
            if self.p >= 0.75 or qubit_errors_change <= 0:
                self.toric.qubit_matrix = new_matrix
                continue
            # acceptence ratio
            if rand.random() < ((self.p / 3.0) / (1.0 - self.p)) ** qubit_errors_change:
                self.toric.qubit_matrix = new_matrix
    
    # plot toric code
    def plot(self, name, eq_class=None):
        self.toric.syndrom('next_state')
        self.toric.plot_toric_code(self.toric.next_state, name, eq_class)


# This is the object we crate to read a file during training
class MCMCDataReader:  
    def __init__(self, file_path, size):
        # file_path needs to be dataframe in pickle format
        self.__file_path = file_path
        # size is the size of the toric code 
        self.__size = size
        self.__df = pd.read_pickle(file_path)
        try:
            self.__df = pd.read_pickle(file_path)
            self.__capacity = self.__df.index[-1][0] + 1  # The number of data samples in the dataset
        except:
            print('No input file for MCMCDataReader')
        self.__current_index = 0
        
    # return next errorchain,distribution pair
    def next(self):
        if self.__current_index < self.__capacity:
            qubit_matrix = self.__df.loc[self.__current_index].loc[0:1].to_numpy(copy=True).reshape((2, self.__size, self.__size))
            eq_distr = self.__df.loc[self.__current_index].loc[2:17].to_numpy(copy=True).reshape((-1))  # kanske inte behöver kopiera här?
            self.__current_index += 1
            return qubit_matrix, eq_distr
        else:
            return None, None  # How do we do this nicely? Maybe it can wrap around?

    def has_next(self):
        return self.__current_index < self.__capacity

    def current_index(self):
        return self.__current_index

    def get_capacity(self):
        return self.__capacity


# parallel tempering method. returns equivalence class distribution distr 
# and the number of steps taken when convergegence is reached, count
def parallel_tempering(init_toric, Nc=None, p=0.1, SEQ=5, TOPS=10, tops_burn=2, eps=0.001, n_tol=1e-4, steps=1000000, iters=10, conv_criteria='error_based'):
    size = init_toric.system_size
    Nc = Nc or size

    # create the diffrent chains in an array
    # number of chains in ladder, must be odd
    if not Nc % 2:
        print('Number of chains was not odd.')

    if tops_burn >= TOPS:
        print('tops_burn has to be smaller than TOPS')

    ladder = []  # ladder to store all chains
    p_end = 0.75  # p at top chain as per high-threshold paper
    tops0 = 0
    resulting_burn_in = 0
    since_burn = 0
    nbr_errors_bottom_chain = np.zeros(steps)
    eq = np.zeros([steps, 16], dtype=np.uint32)  # list of class counts after burn in

    # Save statistics
    qubitlist = []

    # used in error_based/majority_based instead of setting tops0 = TOPS
    tops_change = 0

    convergence_reached = False

    # add and copy state for all chains in ladder
    for i in range(Nc):
        p_i = p + ((p_end - p) / (Nc - 1)) * i
        ladder.append(Chain(size, p_i))
        ladder[i].toric = copy.deepcopy(init_toric)  # give all the same initial state
    ladder[Nc - 1].p_logical = 0.5  # set probability of application of logical operator in top chain

    count = -1
    for j in range(steps):
        # run mcmc for each chain [steps] times
        for i in range(Nc):
            ladder[i].update_chain(iters)
        # current_eq attempt flips from the top down
        ladder[-1].flag = 1
        for i in reversed(range(Nc - 1)):
            if r_flip(ladder[i].toric.qubit_matrix, ladder[i].p, ladder[i + 1].toric.qubit_matrix, ladder[i + 1].p):
                ladder[i].toric, ladder[i + 1].toric = ladder[i + 1].toric, ladder[i].toric
                ladder[i].flag, ladder[i + 1].flag = ladder[i + 1].flag, ladder[i].flag
        if ladder[0].flag == 1:
            tops0 += 1
            ladder[0].flag = 0

        current_eq = define_equivalence_class(ladder[0].toric.qubit_matrix)

        if tops0 >= tops_burn:
            since_burn = j - resulting_burn_in

            eq[since_burn] = eq[since_burn - 1]
            eq[since_burn][current_eq] += 1
            nbr_errors_bottom_chain[since_burn] = np.count_nonzero(ladder[0].toric.qubit_matrix)

        else:
            # number of steps until tops0 = 2
            resulting_burn_in += 1

        if not convergence_reached and tops0 >= TOPS and not since_burn % 10:
            if conv_criteria == 'error_based':
                tops_accepted = tops0 - tops_change
                accept, convergence_reached = conv_crit_error_based(nbr_errors_bottom_chain, since_burn, tops_accepted, SEQ, eps)
                if not accept:
                    tops_change = tops0

        if convergence_reached:
            count=j
            break

        qubitlist.append(ladder[0].toric.qubit_matrix)

    distr = (np.divide(eq[since_burn], since_burn + 1) * 100).astype(np.uint8)
    return distr, count, qubitlist


# convergence criteria used in paper and called ''felkriteriet''
def conv_crit_error_based(nbr_errors_bottom_chain, since_burn, tops_accepted, SEQ, eps):  # Konvergenskriterium 1 i papper
    # last nonzero element of nbr_errors_bottom_chain is since_burn. Length of nonzero part is since_burn + 1
    l = since_burn + 1
    # Calculate average number of errors in 2nd and 4th quarter
    Average_Q2 = np.average(nbr_errors_bottom_chain[(l // 4): (l // 2)])
    Average_Q4 = np.average(nbr_errors_bottom_chain[(3 * l // 4): l])

    # Compare averages
    error = abs(Average_Q2 - Average_Q4)

    if error < eps:
        return True, tops_accepted >= SEQ
    else:
        return False, False


@jit(nopython=True) # @jit needed for numba, r_flip calculates the quotient called r_flip
def r_flip(qubit_lo, p_lo, qubit_hi, p_hi):
    ne_lo = 0
    ne_hi = 0
    for i in range(2):
        for j in range(qubit_lo.shape[1]):
            for k in range(qubit_lo.shape[1]):
                if qubit_lo[i, j, k] != 0:
                    ne_lo += 1
                if qubit_hi[i, j, k] != 0:
                    ne_hi += 1
    # compute eqn (5) in high threshold paper
    if rand.random() < ((p_lo / p_hi) * ((1 - p_hi) / (1 - p_lo))) ** (ne_hi - ne_lo):
        return True
    return False


# applies a random logical operator and returns the resulting qubit_matrix
#  and the change in the total number of error
@jit(nopython=True)
def apply_random_logical(qubit_matrix):
    size = qubit_matrix.shape[1]

    # operator to use, 2 (Y) will make both X and Z on the same layer. 0 is identity
    # one operator for each layer
    operators = [int(rand.random() * 4), int(rand.random() * 4)]

    # ok to not copy, since apply_logical doesnt change input
    result_qubit_matrix = qubit_matrix
    result_error_change = 0

    for layer, op in enumerate(operators):
        if op == 1 or op == 2:
            X_pos = int(rand.random() * size)
        else:
            X_pos = 0
        if op == 3 or op == 2:
            Z_pos = int(rand.random() * size)
        else:
            Z_pos = 0
        result_qubit_matrix, tmp_error_change = apply_logical(result_qubit_matrix, op, layer, X_pos, Z_pos)
        result_error_change += tmp_error_change

    return result_qubit_matrix, result_error_change


@jit(nopython=True)
def apply_logical(qubit_matrix, operator=int, layer=int, X_pos=0, Z_pos=0):
    # Have to make copy, else original matrix is changed
    result_qubit_matrix = np.copy(qubit_matrix)

    # Operator is zero means identity, no need to keep going
    if operator == 0:
        return result_qubit_matrix, 0

    size = qubit_matrix.shape[1]

    error_count = 0

    # layer 0 is qubits on vertical grid lines
    # layer 1 is qubits on horizontal grid lines
    # logical X works orthogonal to grid lines
    # logical Z works parallel to grid lines

    # Transpose copied matrix if layer is 1. Makes next step more straightforward
    # Editing orient_result changes result_qubit matrix whether transposed or not
    if layer == 0:
        orient_result = result_qubit_matrix
    elif layer == 1:
        orient_result = result_qubit_matrix.transpose(0, 2, 1)

    do_X = (operator == 1 or operator == 2)
    do_Z = (operator == 3 or operator == 2)

    # Helper function
    def qubit_update(row, col, op):
        old_qubit = orient_result[layer, row, col]
        new_qubit = rule_table[op][old_qubit]
        orient_result[layer, row, col] = new_qubit
        if old_qubit and not new_qubit:
            return -1
        elif new_qubit and not old_qubit:
            return 1
        else:
            return 0

    for index in range(size):
        if do_X:
            error_count += qubit_update(X_pos, index, 1)
        if do_Z:
            error_count += qubit_update(index, Z_pos, 3)
    return result_qubit_matrix, error_count


@jit(nopython=True)
def apply_stabilizer(qubit_matrix, row=int, col=int, operator=int):
    # gives the resulting qubit error matrix from applying (row, col, operator) stabilizer
    # doesn't update input qubit_matrix
    size = qubit_matrix.shape[1]
    if operator == 1:
        qubit_matrix_layers = np.array([1, 1, 0, 0])
        rows = np.array([row, row, row, (row - 1) % size])
        cols = np.array([col, (col - 1) % size, col, col])

    elif operator == 3:
        qubit_matrix_layers = np.array([1, 0, 0, 1])
        rows = np.array([row, row, row, (row + 1) % size])
        cols = np.array([col, col, (col + 1) % size, col])

    # Have to make copy, else original matrix is changed
    result_qubit_matrix = np.copy(qubit_matrix)
    error_count = 0

    for i in range(4):
        old_qubit = qubit_matrix[qubit_matrix_layers[i], rows[i], cols[i]]
        new_qubit = rule_table[operator][old_qubit]
        result_qubit_matrix[qubit_matrix_layers[i], rows[i], cols[i]] = new_qubit
        if old_qubit and not new_qubit:
            error_count -= 1
        elif new_qubit and not old_qubit:
            error_count += 1

    return result_qubit_matrix, error_count


@jit(nopython=True)
def apply_random_stabilizer(qubit_matrix):
    # select random coordinates where to apply operator
    size = qubit_matrix.shape[1]
    row = int(rand.random() * size)
    col = int(rand.random() * size)
    operator = int(rand.random() * 2)  # we only care about X and Z, and Y is represented by 2. Therefore:
    if operator == 0:
        operator = 3
    return apply_stabilizer(qubit_matrix, row, col, operator)


# applies stabilizers with probability p on each individual plaquette and vertex
def apply_stabilizers_uniform(qubit_matrix, p=0.5):
    size = qubit_matrix.shape[1]
    result_qubit_matrix = np.copy(qubit_matrix)
    random_stabilizers = np.random.rand(2, size, size)
    random_stabilizers = np.less(random_stabilizers, p)

    # Numpy magic for iterating through matrix
    it = np.nditer(random_stabilizers, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            op, row, col = it.multi_index
            if op == 0:
                op = 3
            result_qubit_matrix, _ = apply_stabilizer(result_qubit_matrix, row, col, op)
        it.iternext()
    return result_qubit_matrix


def define_equivalence_class(qubit_matrix):
    # checks odd and even errors in each layer
    # gives a combination of four numbers corresponding to an equivalence class

    # checks odd or even x-errors in first layer
    x1prov = np.count_nonzero(qubit_matrix[0] == 1)

    # checks odd or even z-errors in first layer
    z1prov = np.count_nonzero(qubit_matrix[0] == 3)

    # checks odd or even y-erros in first layer and adds them to total number of x/z errors in first layer
    y1prov = np.count_nonzero(qubit_matrix[0] == 2)
    x1 = x1prov + y1prov
    z1 = z1prov + y1prov

    # checks odd or even x-errors in second layer
    x2prov = np.count_nonzero(qubit_matrix[1] == 1)

    # checks odd or even z-errors in second layer
    z2prov = np.count_nonzero(qubit_matrix[1] == 3)

    # checks odd or even y-erros in second layer and adds them to total number of x/z errors in second layer
    y2prov = np.count_nonzero(qubit_matrix[1] == 2)
    x2 = x2prov + y2prov
    z2 = z2prov + y2prov

    # stores whether there was an odd or even number of errors
    x1 = x1 % 2
    z1 = z1 % 2

    x2 = x2 % 2
    z2 = z2 % 2

    return x1 + z1 * 2 + x2 * 4 + z2 * 8
