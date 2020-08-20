import numpy as np
import copy
import collections
from multiprocessing import Pool
from numba import jit, njit
#from src.toric_model import Toric_code
from src.planar_model import *
from src.util import *
from src.mcmc import *
from src.mwpm import *
#from src.mcmc import MCMCDataReader
from src.mcmc import Chain
from math import log, exp
from operator import itemgetter

NUM_POINTS = 100 #number of datapoints on X axis of histogram

def STDC_rain_hist(init_code, params):
    size = int(params['size'])
    p_error = params['p']
    p_sampling = params['p_sampling']
    droplets = int(params['raindrops'])
    steps = int(params['steps'])*droplets

    # set p_sampling equal to p_error by default
    p_sampling = p_sampling or p_error

    if type(init_code) == list:

        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        eq_chains = [Chain(params['size'], p_sampling, copy.deepcopy(code)) for code in init_code]
        # don't apply uniform stabilizers if low energy inits are provided
        randomize = False
        mwpm_distr = [eq_chains[eq].code.count_errors() for eq in range(nbr_eq_classes)]
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

    num_points = NUM_POINTS

    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = [[{} for _ in range(num_points)] for _ in range(nbr_eq_classes)]
    eqdistr = np.zeros((nbr_eq_classes, num_points))
    beta = -log((p_error / 3) / (1 - p_error))


    with Pool(droplets*nbr_eq_classes) as pool:
        output = pool.map(STDC_fast_droplet, [(copy.deepcopy(eq_chains[eq]), int(steps), randomize, eq, drop, num_points) for drop in range(droplets) for eq in range(nbr_eq_classes)])
    for thread in output:
        samples, eq, _ = thread
        for stage in range(num_points):
            qubitlist[eq][stage].update(samples[stage])

    for eq in range(nbr_eq_classes):
        for stage in range(num_points):
            for key in qubitlist[eq][stage]:
                eqdistr[eq, stage] += exp(-beta * qubitlist[eq][stage][key])

    if randomize == False:
        eqdistr = np.insert(eqdistr, 0, mwpm_distr, axis=1) #add the mwpm solution to the start of the histogram

    return eqdistr

#Number of threads = number of equivalence classes * number of raindrops
def STDC_fast_droplet(input_data_tuple):
    chain, steps, randomize, eq, drop, num_points = input_data_tuple
    samples = [{} for _ in range(num_points)]
    # Start in high energy state
    if randomize == True:
        chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()
    # Do the metropolis steps and add to samples if new chains are found
    for stage in range(num_points):
        for step in range(int(steps/num_points)):
            chain.update_chain(5)
            key = chain.code.qubit_matrix.astype(np.uint8).tostring()
            if key not in samples[stage]:
                for sample in samples[stage:]:
                    sample[hash(key)] = chain.code.count_errors()
    return (samples, eq, drop)

if __name__ == '__main__':
    #benchmark script
    params = {'size': int(5),
              'p': 0.13,
              'Nc': 9,
              'steps': 10, #int(5*int(array_id)**4/100)*100, #int((20000 * (int(array_id)/5)**4)/100)*100, Needs to divide number of data poins
              'iters': 10,
              'conv_criteria': 'error_based',
              'SEQ': 7,
              'TOPS': 10,
              'eps': 0.005,
              'p_sampling': 0.25,
              'raindrops': 4,
              'logical_error_limit': 2000,
              'nbr_eq_classes': 4}
    init_code = Planar_code(int(params['size']))
    init_code.generate_random_error(params['p'])
    init_code = class_sorted_mwpm(init_code)
    for i in range(10):
        STDC_rain_hist(init_code, params)
        print('generated point', ' ', i)
