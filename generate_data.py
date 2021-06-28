import copy

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
from src.mcmc import *
from decoders import *
from src.mwpm import *


# This function generates training data with help of the MCMC algorithm
def generate(params):

    PATH = 'dicts/MWPMp05d7/'

    pickle_defects = open(PATH+"/dict.defects","w")
    pickle_defects.close()

    ####
    pickle_eq_dist = open(PATH+"/dict.eq_distr","w")
    pickle_eq_dist.close()
    

    # Loop to generate data points
    seen = set()
    found_unique = [0,0,0,0]
    while min(found_unique) < 10000:

        # Initiate code
        init_code = Planar_code(params['size'])
        init_code.generate_random_error(params['p_error'])

        init_code.syndrom()
        h = hash(init_code.plaquette_defects.tostring() + init_code.vertex_defects.tostring())
        if h in seen:
            continue
        else:
            seen.add(h)

        # Flatten initial qubit matrix to store in dataframe
        df_qubit = copy.deepcopy(init_code.qubit_matrix)
        eq_true = init_code.define_equivalence_class()


        # Generate data for DataFrame storage  OBS now using full bincount, change this
        choice = regular_mwpm(copy.deepcopy(init_code))
        df_eq_distr = np.zeros((4)).astype(np.uint8)
        df_eq_distr[choice] = 100

        found_unique[choice] += 1
        print(found_unique)


        vertex_defects = init_code.vertex_defects
        plaquette_defects = init_code.plaquette_defects
        
        vertex_defects = 1*vertex_defects
        plaquette_defects = 1*plaquette_defects
        
        pickle_defects = open(PATH+"dict.defects","ab")
        pickle.dump((vertex_defects,plaquette_defects),pickle_defects)
        pickle_defects.close()


        ####
        pickle_eq_dist = open(PATH+"dict.eq_distr","ab")
        pickle.dump(df_eq_distr,pickle_eq_dist)
        pickle_eq_dist.close()
        #####



if __name__ == '__main__':
    # Get job array id, working directory
    job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    array_id = os.getenv('SLURM_ARRAY_TASK_ID')
    local_dir = os.getenv('TMPDIR')

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
