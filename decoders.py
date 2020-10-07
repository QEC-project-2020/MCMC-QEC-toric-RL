import numpy as np
import random as rand
import copy
import collections
import time

from numba import jit, njit
from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.mcmc import *
from src.mwpm import class_sorted_mwpm, regular_mwpm
import pandas as pd
import time

from math import log, exp
from multiprocessing import Pool
from operator import itemgetter


# Original MCMC Parallel tempering method as descibed in high threshold paper
# Parameters also adapted from that paper.
# steps has an upper limit on 50 000 000, which should not be met during operation
def PTEQ(init_code, p, Nc=None, SEQ=2, TOPS=10, tops_burn=2, eps=0.1, steps=50000000, iters=10, conv_criteria='error_based'):
    # either 4 or 16 depending on choice of code topology
    nbr_eq_classes = init_code.nbr_eq_classes

    # If not specified, use size as per paper
    Nc = Nc or init_code.system_size

    # Warn about incorrect parameter inputs
    if tops_burn >= TOPS:
        print('tops_burn has to be smaller than TOPS')

    # initialize variables
    since_burn = 0
    resulting_burn_in = 0
    nbr_errors_bottom_chain = np.zeros(steps)

    # list of class counts after burn in
    eq = np.zeros([steps, nbr_eq_classes], dtype=np.uint32)

    # used in error_based/majority_based instead of setting tops0 = TOPS
    conv_start = 0
    conv_streak = 0

    # Convergence flag
    convergence_reached = False

    # initialize ladder of chains sampled at different temperatures
    ladder = Ladder(p, init_code, Nc, 0.5)

    # Main loop that runs until convergence or max steps (steps) are reached
    for step in range(steps):
        # run metropolis on every chain and perform chain swaps
        ladder.step(iters)

        # Get sample from eq-class of chain in lowest layer of ladder
        current_eq = ladder.chains[0].code.define_equivalence_class()

        # Start saving stats once burn-in period is over
        if ladder.tops0 >= tops_burn:
            since_burn = step - resulting_burn_in

            eq[since_burn] = eq[since_burn - 1]
            eq[since_burn][current_eq] += 1
            nbr_errors_bottom_chain[since_burn] = ladder.chains[0].code.count_errors()
        else:
            # number of steps until tops0 = 2
            resulting_burn_in += 1

        # Check for convergence every 10 samples if burn-in period is over (and conv-crit is set)
        if conv_criteria == 'error_based' and ladder.tops0 >= TOPS:
            accept, convergence_reached = conv_crit_error_based_PT(nbr_errors_bottom_chain, since_burn, conv_streak, SEQ, eps)
            if accept:
                if convergence_reached:
                    break
                conv_streak = ladder.tops0 - conv_start
            else:
                conv_streak = 0
                conv_start = ladder.tops0
    
    # print warning if loop is exited without convergence
    else:
        if conv_criteria == 'error_based':
            print('\n\nWARNING: PTEQ hit max number of steps before convergence:\t', step + 1, '\n\n')

    return (np.divide(eq[since_burn], since_burn + 1) * 100).astype(np.uint8)


# convergence criteria used in paper and called ''felkriteriet''
def conv_crit_error_based_PT(nbr_errors_bottom_chain, since_burn, tops_accepted, SEQ, eps):
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


def single_temp(init_code, p, max_iters):
    # check if init_code is provided as a list of inits for different classes
    if type(init_code) == list:
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init per class is provided
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        # initiate ladder
        chains = [Chain(p, copy.deepcopy(code)) for code in init_code]

    # if init_code is a single code, inits for every class have to be generated
    else:
        nbr_eq_classes = init_code.nbr_eq_classes
        chains = [None] * nbr_eq_classes # list of chain objects
        for eq in range(nbr_eq_classes):
            chains[eq] = Chain(p, copy.deepcopy(init_code))
            chains[eq].code.qubit_matrix = chains[eq].code.to_class(eq) # apply different logical operator to each chain

    nbr_errors_chain = np.zeros((nbr_eq_classes, max_iters))
    mean_array = np.zeros(nbr_eq_classes, dtype=float)

    for eq in range(nbr_eq_classes):
        for j in range(max_iters):
            chains[eq].update_chain_fast(5)
            nbr_errors_chain[eq ,j] = chains[eq].code.count_errors()
            if j == max_iters-1:
                mean_array[eq] = np.average(nbr_errors_chain[eq ,:j])

    return mean_array


def PTDC_droplet(ladder, steps, iters, conv_mult):
    samples = {}
    shortest = 2 * ladder.chains[0].code.system_size ** 2
    conv_step = None
    for step in range(steps):
        # do mcmc iterations and perform swaps
        ladder.step(iters)
        # iterate through chain ladder, record observations
        for chain in ladder.chains:
            # hash error chain for reduced memory usage
            key = hash(chain.code.qubit_matrix.tobytes())
            # check if the error chain has been seen before
            if not key in samples:
                # otherwise, calculate its length
                length = chain.code.count_errors()
                samples[key] = length
                
                # if new shortest chain found, extend sampling time
                if conv_mult and length <= shortest:
                    shortest = length
                    stop = step * conv_mult

        # if no new shortest chain found, end sampling
        if conv_mult and step >= stop and step * 100 >= steps:
            conv_step = step
            break

    return samples#, conv_step


def PTDC(init_code, p_error, p_sampling=None, droplets=4, Nc=None, steps=20000, conv_mult=0):
    p_sampling = p_sampling or p_error
    iters = 10

    if type(init_code) == list:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        # store system_size for brevity
        size = init_code[0].system_size
        # if Nc is not provided, use code system_size
        Nc = Nc or size
        # initiate class ladders
        eq_ladders = [Ladder(p_sampling, eq_code, Nc) for eq_code in init_code]

    else:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code.nbr_eq_classes
        # store system_size for brevity
        size = init_code.system_size
        # if Nc is not provided, use code system_size
        Nc = Nc or size
        # convert init_code to ecery class and initiate ladders
        eq_ladders = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            eq_code = copy.deepcopy(init_code)
            eq_code.qubit_matrix = eq_code.to_class(eq)
            eq_ladders[eq] = Ladder(p_sampling, eq_code, Nc)
        
    # reduce number of steps to account for parallel markov chains
    steps = steps // Nc
    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = {}
    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)
    # keep track of convergence
    conv_step = np.zeros(nbr_eq_classes)
    # keep track of shortest observed chains
    shortest = np.ones(nbr_eq_classes) * (2 * size ** 2)
    # keep track of when to stop if using convergence criteria
    stop = steps
    # error-model
    beta = -log((p_error / 3) / (1 - p_error))

    # initiate worker pool
    if droplets > 1:
        pool = Pool(droplets)

    # Do mcmc sampling, one class at a time
    for eq in range(nbr_eq_classes):
        if droplets == 1:
            qubitlist = PTDC_droplet(eq_ladders[eq], steps, iters, conv_mult)
        else:
            args = [(copy.deepcopy(eq_ladders[eq]), steps, iters, conv_mult) for _ in range(droplets)]
            output = pool.starmap_async(PTDC_droplet, args).get()
            for res in output:
                qubitlist.update(res)

        # mcmc sampling for class is finished. calculate boltzmann factor
        for key in qubitlist:
            eqdistr[eq] += exp(-beta * qubitlist[key])
        qubitlist.clear()

    # Retrun normalized eq_distr
    return (np.divide(eqdistr, sum(eqdistr)) * 100).astype(np.uint8)#, conv_step


def STDC_droplet(chain, steps, randomize, conv_mult):
    # All unique chains will be saved in samples
    samples = {}

    # Convergence variables
    stop = steps
    shortest = 2 * chain.code.system_size ** 2

    # Start in high energy state
    if randomize:
        chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()

    # Do the metropolis steps and add to samples if new chains are found
    for step in range(int(steps)):
        chain.update_chain_fast(5)
        key = hash(chain.code.qubit_matrix.tobytes())
        if key not in samples:
            length = chain.code.count_errors()
            samples[key] = length

            # if new shortest chain found, extend sampling time
            if conv_mult and length <= shortest:
                shortest = length
                stop = step * conv_mult
        
        # if no new shortest chain found, end sampling
        if conv_mult and step >= stop and step * 100 >= steps:
            break

    return samples


def STDC(init_code, p_error, p_sampling=None, droplets=10, steps=20000, conv_mult=0):
    # set p_sampling equal to p_error by default
    p_sampling = p_sampling or p_error

    if type(init_code) == list:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        eq_chains = [Chain(p_sampling, copy.deepcopy(code)) for code in init_code]
        # don't apply uniform stabilizers if low energy inits are provided
        randomize = False

    else:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code.nbr_eq_classes
        # Create chain with p_sampling, this is allowed since N(n) is independet of p.
        eq_chains = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            eq_chains[eq] = Chain(p_sampling, copy.deepcopy(init_code))
            eq_chains[eq].code.qubit_matrix = eq_chains[eq].code.to_class(eq)
        # apply uniform stabilizers, i.e. rain
        randomize = True

    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = {}

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)

    # error-model
    beta = -log((p_error / 3) / (1 - p_error))

    if droplets > 1:
        pool = Pool(droplets)

    for eq in range(nbr_eq_classes):
        # go to class eq and apply stabilizers
        chain = eq_chains[eq]

        if droplets == 1:
            qubitlist = STDC_droplet(copy.deepcopy(chain), steps, randomize, conv_mult)
        else:
            args = [(copy.deepcopy(chain), steps, randomize, conv_mult) for _ in range(droplets)]
            output = pool.starmap_async(STDC_droplet, args).get()
            for res in output:
                qubitlist.update(res)

        # compute Z_E
        for key in qubitlist:
            eqdistr[eq] += exp(-beta * qubitlist[key])
        qubitlist.clear()

    # Retrun normalized eq_distr
    return (np.divide(eqdistr, sum(eqdistr)) * 100)


def PTRC_droplet(ladder, steps, iters, conv_mult):

    shortest = 2 * ladder.chains[0].code.system_size ** 2

    # ladder of lengths of qubit matrices
    unique_lengths_ladder = [{} for _ in range(ladder.Nc)]
    # ladder of observations of qubit matrix lengths
    len_counts_ladder = [{} for _ in range(ladder.Nc)]

    for step in range(steps):
        # do mcmc iterations and perform swaps
        ladder.step(iters)
        # iterate through ladder, recording chains observations
        for i, chain in enumerate(ladder.chains):
            unique_lengths = unique_lengths_ladder[i]
            len_counts = len_counts_ladder[i]
            # hash qubit matrix to reduce memory usage
            key = hash(chain.code.qubit_matrix.tobytes())
            # check if chain has been observed before
            if key in unique_lengths:
                # if so, read it's length from dict
                length = unique_lengths[key]
                # increment m(n)
                len_counts[length][1] += 1

            else:
                # otherwise, calculate length
                length = chain.code.count_errors()
                # update dict of observed chains
                unique_lengths[key] = length
                # check if chain of this length was observed before
                if length in len_counts:
                    # increment N(n)
                    len_counts[length][0] += 1
                    # increment m(n)
                    len_counts[length][1] += 1

                else:
                    # initialize N(n) and m(n)
                    len_counts[length] = [1, 1]

                # if new shortest chain found, extend sampling time
                if conv_mult and length <= shortest:
                    shortest = length
                    stop = step * conv_mult
        
        ## if no new shortest chain found, end sampling
        #if conv_mult and step >= stop and step * 100 >= steps:
        #    conv_step = step
        #    break
    
    return unique_lengths_ladder, len_counts_ladder


def PTRC(init_code, p_error, p_sampling=None, droplets=4, Nc=None, steps=20000, conv_mult=2.0):
    p_sampling = p_sampling or p_error
    iters = 10

    if type(init_code) == list:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        # store system_size for brevity
        size = init_code[0].system_size
        # if Nc is not provided, use code system_size
        Nc = Nc or size
        # initiate class ladders
        eq_ladders = [Ladder(p_sampling, eq_code, Nc) for eq_code in init_code]

    else:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code.nbr_eq_classes
        # store system_size for brevity
        size = init_code[0].system_size
        # if Nc is not provided, use code system_size
        Nc = Nc or size
        # convert init_code to every class and initiate ladders
        eq_ladders = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            eq_code = copy.deepcopy(init_code)
            eq_code.qubit_matrix = eq_code.to_class(eq)
            eq_ladders[eq] = Ladder(p_sampling, eq_code, Nc)

    # reduce number of steps to account for parallel markov chains
    steps = steps // Nc
    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = {}
    # keep track of convergence
    conv_step = np.zeros(nbr_eq_classes)
    # keep track of shortest observed chains
    shortest = np.ones(nbr_eq_classes) * (2 * size ** 2)
    # keep track of when to stop if using convergence criteria
    stop = steps
    # inverse temperature when writing probability in exponential form
    beta_error = -log((p_error / 3) / (1 - p_error))
    # array of betas correspoding to ladder temperatures
    beta_ladder = -np.log((eq_ladders[0].p_ladder[:-1] / 3) / (1 - eq_ladders[0].p_ladder[:-1]))
    d_beta = beta_ladder - beta_error
    # Array to hold the boltzmann factors for every class
    Z_arr = np.zeros(nbr_eq_classes)

    # initiate worker pool
    if droplets > 1:
        pool = Pool(droplets)

    # do mcmc sampling, one class at a time
    for eq in range(nbr_eq_classes):

        if droplets == 1:
            unique_lengths_ladder, len_counts_ladder = PTRC_droplet(eq_ladders[eq], steps, iters, conv_mult)
        else:
            # ladder of lengths of qubit matrices
            unique_lengths_ladder = [{} for _ in range(Nc)]
            # ladder of observations of qubit matrix lengths
            len_counts_ladder = [{} for _ in range(Nc)]

            args = [(copy.deepcopy(eq_ladders[eq]), steps, iters, conv_mult) for _ in range(droplets)]
            output = pool.starmap_async(PTRC_droplet, args).get()

            # combine outputs
            for res in output:
                # iterate ladder
                for i in range(Nc):
                    unique_lengths_ladder[i].update(res[0][i])
                    # iterate output unique_lengths dictionary
                    for length, counts in res[1][i].items():
                        # aggregate or init length counter
                        if length in len_counts_ladder[i]:
                            # aggregate N(n)
                            len_counts_ladder[i][length][0] += counts[0]
                            # aggregate m(n)
                            len_counts_ladder[i][length][1] += counts[1]
                        else:
                            len_counts_ladder[i][length] = counts
        
        # iterate through all but top chain in ladder
        for i in range(Nc - 1):
            # sort len_counts by length
            sorted_counts = sorted(len_counts_ladder[i].items(), key=itemgetter(0))
            # make length and count array from sorted list
            lengths, counts = [np.array(lst) for lst in zip(*sorted_counts)]
            
            ## calculate C estimate for each length, count pair
            #C_ests = counts[:, 0] / counts[:, 1] * np.exp(-beta_ladder[i] * (lengths - lengths[0]))
            ## remove outlier estimates
            #tmp = C_ests[C_ests * 2 > C_ests[0]]
            ## calculate final estimate
            #C_mean = np.sqrt(np.mean(np.square(tmp))) # Root mean square so the average is "top-weighted"
            
            C_mean = np.mean(counts[:2, 0] / counts[:2, 1] * np.exp(-beta_ladder[i] * (lengths[:2] - lengths[0])))

            # calculate boltzmann factor from C estimate
            Z_est = C_mean * (counts[:, 1] * np.exp(lengths * d_beta[i] - beta_ladder[i] * lengths[0])).sum()
            # Accumulate boltzmann factor for equivalence class
            Z_arr[eq] += Z_est

    # Retrun normalized eq_distr
    return (Z_arr / np.sum(Z_arr) * 100).astype(np.uint8)#, conv_step


def STRC_droplet(chain, steps, max_length, eq, randomize, conv_mult):
    unique_lengths = {}
    len_counts = {}

    # List of unique shortest and next shortets chains
    short_unique = [{'temp': max_length} for _ in range(2)]

    # Variables to easily keep track of the length of chains in short_unique
    shortest = max_length
    next_shortest = max_length

    # Convergence variable
    stop = steps
    
    # Apply random stabilizers to start in high temperature state
    if randomize:
        chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()

    # Generate chains
    for step in range(steps):
        # Do metropolis sampling
        chain.update_chain_fast(5)
        # Convert the current qubit matrix to string for hashing
        key = hash(chain.code.qubit_matrix.tobytes())

        # Check if this error chain has already been seen by comparing hashes
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

                # Check if this chain is same length as previous shortest chain
                if length == shortest:
                    # Then add it to the set of seen short chains
                    short_unique[0][key] = length

                    # if new shortest chain found, extend sampling time
                    stop = step * conv_mult if conv_mult else stop

                # Otherwise, check if this chain same length as previous next shortest chain
                elif length == next_shortest:
                    # Then add it to the set of seen next shortest chains
                    short_unique[1][key] = length

            else:
                #print('it has a new length!')
                # Initiate counter for chains of this length
                len_counts[unique_lengths[key]] = 1
                # Check if this chain is shorter than prevous shortest chain
                if length < shortest:
                    # Then the previous shortest length is the new next shortest
                    next_shortest = shortest
                    shortest = length
                    
                    # Clear next shortest set and set i equal to shortest
                    short_unique[1].clear()
                    short_unique[1].update(short_unique[0])
                    # And the current length is the new shortest
                    short_unique[0].clear()
                    short_unique[0][key] = length

                    # if new shortest chain found, extend sampling time
                    stop = step * conv_mult if conv_mult else stop
                
                # Otherwise, check if this chain is shorter than previous next shortest chain
                elif length < next_shortest:
                    # Then reset stats of next shortest chain
                    next_shortest = length

                    # Clear and update next shortest set
                    short_unique[1].clear()
                    short_unique[1][key] = length

        # if no new chain found, end sampling
        if conv_mult and step >= stop and step * 100 >= steps:
            break

    return unique_lengths, len_counts, short_unique

 
def STRC(init_code, p_error, p_sampling=None, droplets=10, steps=20000, conv_mult=0):
    # set p_sampling equal to p_error by default
    p_sampling = p_sampling or p_error

    if type(init_code) == list:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        # Create chains with p_sampling, this is allowed since N(n) is independet of p.
        eq_chains = [Chain(p_sampling, copy.deepcopy(code)) for code in init_code]
        # don't apply uniform stabilizers if low energy inits are provided
        randomize = False

    else:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code.nbr_eq_classes
        # Create chains with p_sampling, this is allowed since N(n) is independet of p.
        eq_chains = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            eq_chains[eq] = Chain(p_sampling, copy.deepcopy(init_code))
            eq_chains[eq].code.qubit_matrix = eq_chains[eq].code.to_class(eq)
        # apply uniform stabilizers, i.e. rain
        randomize = True

    # error model
    beta_error = -log((p_error / 3) / (1 - p_error))
    beta_sampling = -log((p_sampling / 3) / (1 - p_sampling))
    d_beta = beta_sampling - beta_error

    # Array to hold the boltzmann factors for every class
    Z_arr = np.zeros(nbr_eq_classes)

    # Largest possible chain length
    max_length = 2 * eq_chains[0].code.system_size ** 2

    if droplets > 1:
        pool = Pool(droplets)

    # Iterate through equivalence classes
    for eq in range(nbr_eq_classes):
        chain = eq_chains[eq]

        # Start parallel processes with droplets.
        if droplets == 1:
            unique_lengths, len_counts, short_unique = STRC_droplet(copy.deepcopy(chain), steps, max_length, eq, randomize, conv_mult)
            shortest = next(iter(short_unique[0].values()))
            next_shortest = next(iter(short_unique[1].values()))
        else:
            args = [(copy.deepcopy(chain), steps, max_length, eq, randomize, conv_mult) for _ in range(droplets)]
            output = pool.starmap_async(STRC_droplet, args).get()

            # We need to combine the results from all raindrops
            unique_lengths = {}
            len_counts = {}
            short_unique = [{} for _ in range(2)]

            shortest = max_length
            next_shortest = max_length

            # Find shortest and next shortest length found by any chain
            for i in range(droplets):
                _,_,data = output[i]
                if next(iter(data[0].values())) < shortest:
                    next_shortest = shortest
                    shortest = next(iter(data[0].values()))
                if next(iter(data[1].values())) < next_shortest:
                    next_shortest = next(iter(data[1].values()))
            
            # Add data from each droplet to the combined dataset
            for i in range(droplets):
                # Unpack results
                unique_lengths_i, len_counts_i, short_unique_i = output[i]
                
                # Combine unique lengths ( not really needed? )
                unique_lengths.update(unique_lengths_i)

                # Combine len_counts
                for key in len_counts_i:
                    if key in len_counts:
                        len_counts[key] += len_counts_i[key]
                    else:
                        len_counts[key] = len_counts_i[key]
                
                # Combine the sets of shortest and next shortest chains
                shortest_i = next(iter(short_unique_i[0].values()))
                next_shortest_i = next(iter(short_unique_i[1].values()))

                if shortest_i == shortest:
                    short_unique[0].update(short_unique_i[0])
                if shortest_i == next_shortest:
                    short_unique[1].update(short_unique_i[0])
                if next_shortest_i == next_shortest:
                    short_unique[1].update(short_unique_i[1])

        # Partial result needed for boltzmann factor
        shortest_count = len(short_unique[0])
        shortest_fraction = shortest_count / len_counts[shortest]

        next_shortest_count = len(short_unique[1])
        
        # Handle rare cases where only one chain length is observed
        if next_shortest != max_length:
            next_shortest_fraction = next_shortest_count / len_counts[next_shortest]
            mean_fraction = 0.5 * (shortest_fraction + next_shortest_fraction * exp(-beta_sampling * (next_shortest - shortest)))
        
        else:
            mean_fraction = shortest_fraction

        # Calculate boltzmann factor from observed chain lengths
        Z_e = sum([m * exp(-beta_sampling * shortest + d_beta * l) for l, m in len_counts.items()]) * mean_fraction
        Z_arr[eq] = Z_e

    # Use boltzmann factors as relative probabilities and normalize distribution
    return (Z_arr / np.sum(Z_arr) * 100)


if __name__ == '__main__':
    size = 3
    steps = 10 * size ** 4
    #reader = MCMCDataReader('data/data_7x7_p_0.19.xz', size)
    p_error = 0.10
    p_sampling = 0.30
    init_code = Planar_code(size)
    tries = 1
    distrs = np.zeros((tries, init_code.nbr_eq_classes))

    #from line_profiler import LineProfiler
    #lp = LineProfiler()
    #lp_wrapper = lp(STRC)

    for i in range(2):
        init_code.generate_random_error(p_error)
        ground_state = init_code.define_equivalence_class()
        print('Ground state:', ground_state)
        
        #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
        #init_qubit = np.copy(init_code.qubit_matrix)

        #class_init = class_sorted_mwpm(init_code)
        #mwpm_init = regular_mwpm(init_code)

        print('################ Chain', i+1 , '###################')
        
        for i in range(tries):
            #t0 = time.time()
            #v1, most_likely_eq, convergece = single_temp(init_code, p=p_error, max_iters=steps, eps=0.005, conv_criteria = None)
            #print('Try single_temp', i+1, ':', v1, 'most_likely_eq', most_likely_eq, 'convergence:', convergece, time.time()-t0)
            t0 = time.time()
            distrs[i] = STDC(copy.deepcopy(init_code), p_error=p_error, p_sampling=p_sampling, steps=steps, droplets=4, conv_mult=0)
            print('Try STDC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'time:', time.time()-t0)
            t0 = time.time()
            distrs[i] = STRC(copy.deepcopy(init_code), p_error=p_error, p_sampling=p_sampling, steps=steps, droplets=4, conv_mult=0)
            print('Try STRC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'time:', time.time()-t0)
            t0 = time.time()
            distrs[i] = PTEQ(copy.deepcopy(init_code), p=p_error)
            print('Try PTEQ       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'time:', time.time()-t0)
            t0 = time.time()
            distrs[i] = PTDC(copy.deepcopy(init_code), p_error=p_error, droplets=4, conv_mult=0)
            print('Try PTDC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'time:', time.time()-t0)
            t0 = time.time()
            distrs[i] = PTRC(copy.deepcopy(init_code), p_error=p_error, droplets=4, conv_mult=0)
            print('Try PTRC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'time:', time.time()-t0)
