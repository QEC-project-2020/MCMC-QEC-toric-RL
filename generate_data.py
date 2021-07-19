import copy
from numba.core.types.npytypes import UnicodeCharSeq

from numpy.lib.arraysetops import unique
from decoders_biasednoise import PTEQ_biased, PTEQ_alpha
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd

from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.xzzx_model import xzzx_code
from src.rotated_surface_model import RotSurCode
from src.mcmc import *
from decoders import *
from src.mwpm import *

import warnings


# This function generates training data with help of the MCMC algorithm
def generate(params):
    
    warnings.filterwarnings("ignore")

    PATH = 'dicts/d7_rot/'

    pickle_defects = open(PATH+"dict.defects","w")
    pickle_defects.close()

    ####
    pickle_eq_dist = open(PATH+"dict.stdc_eq_distr","w")
    pickle_eq_dist.close()

    ####
    pickle_qubitmatrix = open(PATH+"dict.qubitmatrix","w")
    pickle_qubitmatrix.close()
    ####
    pickle_trueeq = open(PATH+"dict.trueeq","w")
    pickle_trueeq.close()
    

    # Loop to generate data points
    seen = set()
    found_unique = [0,0,0,0]
    found = 0

    total = 60000

    unique = True
    balanced = True

    while found < total:

        # Initiate code
        init_code = RotSurCode(params['size'])
        init_code.generate_random_error(params['p_error']/3, params['p_error']/3, params['p_error']/3)

        init_code.syndrome()
        if unique:
            h = hash(init_code.plaquette_defects.tostring())
            if h in seen:
                continue
            else:
                seen.add(h)

        # Flatten initial qubit matrix to store in dataframe
        df_qubit = copy.deepcopy(init_code.qubit_matrix)
        eq_true = init_code.define_equivalence_class()


        # Generate data for DataFrame storage  OBS now using full bincount, change this
        if balanced:
            #choice = regular_mwpm(copy.deepcopy(init_code))
            #df_eq_distr = np.zeros((4)).astype(np.uint8)
            #df_eq_distr[choice] = 100

            if found_unique[eq_true] >= total/4:
                continue

            found_unique[eq_true] += 1
            print(found_unique)
        found += 1


        #vertex_defects = init_code.vertex_defects
        plaquette_defects = init_code.plaquette_defects
        
        #vertex_defects = 1*vertex_defects
        plaquette_defects = 1*plaquette_defects
        
        pickle_defects = open(PATH+"dict.defects","ab")
        pickle.dump((plaquette_defects),pickle_defects)
        pickle_defects.close()

        pickle_qubitmatrix = open(PATH+"dict.qubitmatrix","ab")
        pickle.dump(df_qubit,pickle_qubitmatrix)
        pickle_qubitmatrix.close()


        ## USE STDC to get better guess
        #init_code = class_sorted_mwpm(init_code)
        df_eq_distr = STDC(init_code, params['p_error'], params['p_sampling'], droplets=1, steps=5*params['size']**4)

        ####
        pickle_eq_dist = open(PATH+"dict.stdc_eq_distr","ab")
        pickle.dump(df_eq_distr,pickle_eq_dist)
        pickle_eq_dist.close()
        #####

        pickle_trueeq = open(PATH+"dict.trueeq","ab")
        pickle.dump(eq_true,pickle_trueeq)
        pickle_trueeq.close()
        #####



if __name__ == '__main__':
    # Get job array id, working directory
    job_id = 0#os.getenv('SLURM_ARRAY_JOB_ID')
    array_id =0# os.getenv('SLURM_ARRAY_TASK_ID')
    local_dir = './temp'#os.getenv('TMPDIR')

    params = {'code': "xzzx",
            'method': "MWPM",
            'size': 7,
            'noise': 'alpha',
            'p_error': 0.05,
            'eta': 2,
            'alpha': 2,
            'p_sampling': 0.25,#np.round((0.05 + float(array_id) / 50), decimals=2),
            'droplets':1,
            'mwpm_init':False,
            'fixed_errors':None,
            'Nc':None,
            'iters': 10,
            'conv_criteria': 'error_based',
            'SEQ': 2,
            'TOPS': 10,
            'eps': 0.1}
    # Steps is a function of code size L
    params.update({'steps': int(1e7)})

    print('Nbr of steps to take if applicable:', params['steps'])

    # Generate data
    generate(params)

    # View data file
    
    # iterator = MCMCDataReader(file_path, params['size'])
    # data = iterator.full()
    # for k in range(int(len(data)/2)):
    #     qubit_matrix = data[2*k]#.reshape(2,params['size'],params['size'])
    #     eq_distr = data[2*k+1]

    #     print(qubit_matrix)
    #     print(eq_distr)
