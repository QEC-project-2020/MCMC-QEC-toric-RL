import numpy as np
import random as rand
import copy
import collections


from numba import jit, prange
from src.toric_model import Toric_code
from src.util import Action
from src.mcmc import *
from src.mcmc import Chain
from single_temperature import apply_logical_operator
import pandas as pd
import time

# finds the stabilizers matrices of all the shortest chains
def get_size_shortest_chains(qubit_matrix, size, p, steps=100000, length_order = 0, degeneration = False):
    chain = Chain(size, p)  # this p needs not be the same as p, as it is used to determine how we sample N(n)
    chain.toric.qubit_matrix = qubit_matrix
    qubitlist = []


    #for i in range(16):
    #chain.toric.qubit_matrix = apply_logical_operator(qubit_matrix, i)  # apply different logical operator to each chain
    # We start in a state with high entropy, therefore we let mcmc "settle down" before getting samples.
    for _ in range(int(steps*0.1)):
        chain.update_chain(5)
    for _ in range(int(steps*0.9)):
        chain.update_chain(5)
        qubitlist.append(chain.toric.qubit_matrix)

    # Only consider unique elements
    qubitlist = np.unique(qubitlist, axis=0)

    #error_diff_matrix_list = []
    shortest = 2*size*size+1 # maximum length of a chain + 1
    for i in range(len(qubitlist)):
        shortest = np.minimum(shortest, np.count_nonzero(qubitlist[i]))
    #print(shortest, 'shortest chain length')

    min_diff = np.ones(len(qubitlist))*2*size*size+1 #[size*size]#np.ones(len(qubitlist))*size*size
    prod =  np.zeros((len(qubitlist)))
    for i in range(len(qubitlist)):
        for j in range(len(qubitlist)):
            if j != i:
                xi = np.count_nonzero(qubitlist[i])
                xj = np.count_nonzero(qubitlist[j])
                if xi == xj and xi == shortest + length_order:
                    #print(np.count_nonzero(np.subtract(qubitlist[i], qubitlist[j])), 'np.count_nonzero(np.subtract(qubitlist[i], qubitlist[j]))')
                    #print(min_diff[i], 'min_diff[i]')
                    #print(np.count_nonzero(np.subtract(qubitlist[i], qubitlist[j])), "i = ", i)
                    #print(min_diff[i], 'min_diff i =', i)
                    diff = qubitlist[i] ^ qubitlist[j]

                    not_zero = np.count_nonzero(diff)
                    if min_diff[i] >= not_zero:
                        min_diff[i] = not_zero
                        ''''
                        indices = np.nonzero(diff)
                        for k in range(len(indices)):
                            for n in range(len(indices)):
                                if np.amax(np.absolute(indices[k] - indices[n])) > 1: break
                                else:
                                    prod[i] = np.prod(diff[np.nonzero(diff)])
                                    min_diff[i] =  not_zero
                            else: continue
                            break
                            '''
                    #print(np.nonzero(diff))



                    #print(counter)
                    #print(min_diff[i])
                #matrix = np.absolute(np.subtract(qubitlist[i], qubitlist[j]))
                #if xi == xj:
                #    error_diff_matrix_list.append(matrix)"""

    #print(min_diff[:counter], 'shortest difference to other chain of order ' + str(length_order))

    filter = []
    for element in min_diff:
        if element != 2*size*size+1:
            filter.append(element)

    #print('delta_error', filter)
    if len(filter)>0:
        ratio = np.count_nonzero(np.array(filter) == 4)/len(filter)
    else: ratio = 2

    '''prodfilter = []
    for i in range(len(prod)):
        if prod[i] != 0 and min_diff[i] !=  size*size+1: prodfilter.append(prod[i])
    #print('product_of_error_difference', prodfilter)
    if len(prodfilter) > 0: x_ratio = np.count_nonzero(np.array(prodfilter) == 1) /len(prodfilter)
    if len(prodfilter) > 0: z_ratio =  np.count_nonzero(np.array(prodfilter) == 81) /len(prodfilter)
    if len(prodfilter)  == 0 and degeneration == True: return 2
    elif len(prodfilter)  == 0 and degeneration == False: return 1
    #print('% of 4s: ', x_ratio + z_ratio)
    return x_ratio + z_ratio
    '''
    return ratio

    #return (np.divide(eqdistr, sum(eqdistr)) * 100).astype(np.uint8)
def eq_class_compare(qubit_matrix, size, p, steps=100000, length_order = 0, degeneration = False):
    matrix_lst = []
    for i in range(16):
        matrix_lst.append(apply_logical_operator(qubit_matrix, i))
    result = np.zeros(16)
    for i in range(16):
         result[i] = get_size_shortest_chains(matrix_lst[i], size, p, steps, length_order , degeneration)
    return result


def main1():
    t0 = time.time()
    size = 5
    toric_model = Toric_code(size)
    p_error = 0.15
    iters = 100
    length_order = 0 #zero for shortest chains

    summa = 0
    counter = 0
    degeneration = False #True returns conditional probability given that the chain is degenerate

    for i in range(iters):
        toric_model.generate_random_error(p_error = p_error)
        #print(toric_model.qubit_matrix)
        procent_fours = get_size_shortest_chains(toric_model.qubit_matrix, size = size, p = p_error , steps=5000, length_order =  length_order, degeneration = degeneration)
        if procent_fours == 2:
            counter+=1
            continue
        print("#",i, " %4s: ", procent_fours)
        summa = summa + procent_fours
        #print('runtime ', time.time()-t0, 's')
    print("average % 4s, degeneration == ", degeneration , (summa)/(iters-counter))

def main2():
    for i in range(10):
        size = 5
        toric_model = Toric_code(size)
        p_error = 0.15
        toric_model.generate_random_error(p_error = p_error)
        length_order = 0 #zero for shortest chains
        result = eq_class_compare(toric_model.qubit_matrix, size = size, p = p_error, steps=10000, length_order = 0, degeneration = True)
        print(result)

if __name__ == '__main__':
    main2()
