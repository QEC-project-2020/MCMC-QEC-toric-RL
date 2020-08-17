import copy  # not used
import os
import sys
import time

import numpy as np
import pandas as pd

from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.mcmc import *
from decoders import *
from src.mwpm import *


# This function generates training data with help of the MCMC algorithm
def generate(file_path, params, max_capacity=10**4, nbr_datapoints=10**6, fixed_errors=None):

    if params['code'] == 'planar':
        nbr_eq_class = 4
    elif params['code'] == 'toric':
        nbr_eq_class = 16
    
    if params['method'] == "all":
        nbr_eq_class *= 3
    # Creates data file if there is none otherwise adds to it
    try:
        df = pd.read_pickle(file_path)
        nbr_existing_data = df.index[-1][0] + 1
    except:
        df = pd.DataFrame()
        nbr_existing_data = 0

    print('\nDataFrame with ' + str(nbr_existing_data) +
          ' datapoints opened at: ' + str(file_path))

    # Stop the file from exceeding the max limit of nbr of datapoints
    nbr_to_generate = min(max_capacity-nbr_existing_data, nbr_datapoints)
    if nbr_to_generate < nbr_datapoints:
        print('Generating ' + str(max(nbr_to_generate, 0))
              + ' datapoins instead of ' + str(nbr_datapoints)
              + ', as the given number would overflow existing file')

    if fixed_errors != None:
        nbr_to_generate = 10000000
    failed_syndroms = 0

    df_list = []  # Initiate temporary list

    # Loop to generate data points
    for i in range(nbr_existing_data, nbr_existing_data + nbr_to_generate):
        print('Starting generation of point nr: ' + str(i + 1))

        # Initiate code
        if params['code'] == 'toric':
            init_code = Toric_code(params['size'])
            init_code.generate_random_error(params['p_error'])
        elif params['code'] == 'planar':
            init_code = Planar_code(params['size'])
            init_code.generate_random_error(params['p_error'])
 
        # Flatten initial qubit matrix to store in dataframe
        df_qubit = copy.deepcopy(init_code.qubit_matrix)
        eq_true = init_code.define_equivalence_class()


        
        if params['mwpm_init']: #get mwpm starting points
            init_code = class_sorted_mwpm(init_code)
            print('Starting in MWPM state')
        else: #randomize input matrix, no trace of seed.
            init_code.qubit_matrix, _ = init_code.apply_random_logical()
            init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            print('Starting in random state')

        # Generate data for DataFrame storage  OBS now using full bincount, change this
        if params['method'] == "PTEQ":
            df_eq_distr = PTEQ(init_code, params['p_error'])
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        if params['method'] == "PTDC":
            df_eq_distr, conv = PTDC(init_code, params['p_error'], params['p_sampling'])
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        if params['method'] == "PTRC":
            df_eq_distr, conv = PTRC(init_code, params['p_error'], params['p_sampling'])
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "STDC":
            df_eq_distr = STDC(init_code, params['size'], params['p_error'], params['p_sampling'], steps=params['steps'], droplets=params['droplets'])
            df_eq_distr = np.array(df_eq_distr)
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "ST":
            df_eq_distr = single_temp(init_code, params['p_error'],params['steps'])
            df_eq_distr = np.array(df_eq_distr)
            if np.argmin(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "STRC":
            df_eq_distr = STRC(init_code, params['size'], params['p_error'], p_sampling=params['p_sampling'], steps=params['steps'], droplets=params['droplets'])
            df_eq_distr = np.array(df_eq_distr)
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "all":
            #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr1 = single_temp(init_code, params['p_error'],params['steps'])

            #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr2 = STDC(init_code, params['size'], params['p_error'], p_sampling=params['p_sampling'], steps=params['steps'], droplets=params['droplets'])

            #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr3 = STRC(init_code, params['size'], params['p_error'], p_sampling=params['p_sampling'], steps=params['steps'], droplets=params['droplets'])

            df_eq_distr = np.concatenate((df_eq_distr1,df_eq_distr2,df_eq_distr3), axis=0)
        elif params['method'] == "eMWPM":
            out = class_sorted_mwpm(copy.deepcopy(init_code))
            lens = np.zeros((4))
            for j in range(4):
                lens[j] = out[j].count_errors()
            choice = np.argmin(lens)
            df_eq_distr = np.zeros((4)).astype(np.uint8)
            df_eq_distr[choice] = 100
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "MWPM":
            choice = regular_mwpm(copy.deepcopy(init_code))
            df_eq_distr = np.zeros((4)).astype(np.uint8)
            df_eq_distr[choice] = 100
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1

        # Generate data for DataFrame storage  OBS now using full bincount, change this

        # Create indices for generated data
        names = ['data_nr', 'type']
        index_qubit = pd.MultiIndex.from_product([[i], np.arange(1)],
                                                 names=names)
        index_distr = pd.MultiIndex.from_product([[i], np.arange(1)+1], names=names)

        # Add data to Dataframes
        df_qubit = pd.DataFrame([[df_qubit.astype(np.uint8)]], index=index_qubit,
                                columns=['data'])
        df_distr = pd.DataFrame([[df_eq_distr]],
                                index=index_distr, columns=['data'])

        # Add dataframes to temporary list to shorten computation time
        
        df_list.append(df_qubit)
        df_list.append(df_distr)

        # Every x iteration adds data to data file from temporary list
        # and clears temporary list
        
        if (i + 1) % 10000 == 0: # this needs to be sufficiently big that rsync has time to sync files before update, maybe change this to be time-based instead.
            df = df.append(df_list)
            df_list.clear()
            print('Intermediate save point reached (writing over)')
            df.to_pickle(file_path)
            print('Failed so far:', failed_syndroms)
        
        # If the desired amount of errors have been achieved, break the loop and finish up
        if failed_syndroms == fixed_errors:
            print('Desired amount of failes syndroms achieved, breaking loop.')
            break

    # Adds any remaining data from temporary list to data file when run is over
    if len(df_list) > 0:
        df = df.append(df_list)
        print('\nSaving all generated data (writing over)')
        df.to_pickle(file_path)
    
    print('\nCompleted')


if __name__ == '__main__':
    # Get job array id, working directory
    try:
        array_id = os.getenv('SLURM_ARRAY_TASK_ID')
        local_dir = os.getenv('TMPDIR')
    except:
        array_id = '0'
        local_dir = './data'
        print('Invalid environment variables, using array_id 0 and local dir.')

    params = {'code': "planar",
            'method': "PTRC",
            'size': 15,
            'p_error': np.round((0.05 + float(array_id) / 50), decimals=2),
            'p_sampling': 0.25,#np.round((0.05 + float(array_id) / 50), decimals=2),
            'droplets':1,
            'mwpm_init':True,
            'fixed_errors':None,
            'Nc':None,
            'iters': 10,
            'conv_criteria': 'error_based',
            'SEQ': 2,
            'TOPS': 10,
            'eps': 0.1}
    # Steps is a function of code size L
    params.update({'steps': int(params['size'] ** 4)})

    print('Nbr of steps to take if applicable:', params['steps'])

    # Build file path
    file_path = os.path.join(local_dir, 'data_size_'+str(params['size'])+'_method_'+params['method']+'_id_' + array_id + '_perror_' + str(params['p_error']) + '.xz')

    # Generate data
    generate(file_path, params, nbr_datapoints=10000, fixed_errors=params['fixed_errors'])

    # View data file
    
    '''iterator = MCMCDataReader(file_path, params['size'])
    data = iterator.full()
    for k in range(int(len(data)/2)):
        qubit_matrix = data[2*k].reshape(2,params['size'],params['size'])
        eq_distr = data[2*k+1]

        print(qubit_matrix)
        print(eq_distr)'''
