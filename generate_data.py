import copy  # not used
import os
import sys
import time

import numpy as np
import pandas as pd

from src.toric_model import Toric_code
from src.mcmc import *
from decoders import single_temp_direct_sum, single_temp


# This function generates training data with help of the MCMC algorithm
def generate(file_path, params, timeout,
             max_capacity=10**4, nbr_datapoints=10**6, method="PTEC"):

    t_start = time.time()  # Initiates timing of run

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

    df_list = []  # Initiate temporary list

    # Loop to generate data points
    for i in np.arange(nbr_to_generate) + nbr_existing_data:

        # Breaks if run has exceeded timeout-value
        if time.time() - t_start > timeout:
            print("timeout reached: " + str(timeout) + "s")
            break

        print('Starting generation of point nr: ' + str(i + 1))

        # Initiate toric
        init_toric = Toric_code(params['size'])
        init_toric.generate_random_error(params['p'])

        #randomize input matrix, no trace of seed.
        input_matrix, _ = apply_random_logical(init_toric.qubit_matrix)
        input_matrix = apply_stabilizers_uniform(input_matrix) # fix so all uses these

        # Generate data for DataFrame storage  OBS now using full bincount, change this
        if method == "PTEC":
            df_eq_distr = parallel_tempering(init_toric, params['Nc'],
                                         p=params['p'], steps=params['steps'],
                                         iters=params['iters'],
                                         conv_criteria=params['conv_criteria'])
        elif method == "STDC":
            df_eq_distr = single_temp_direct_sum(input_matrix,params['size'],params['p'])
            df_eq_distr = np.array(df_eq_distr)
        elif method == "ST":
            df_eq_distr = single_temp(init_toric,params['p'],params['steps'], params['eps'])
            df_eq_distr = np.array(df_eq_distr)
        else:
            raise ValueError('Invalid method, use "PTEC", "STDC" or "ST".')

        # Generate data for DataFrame storage  OBS now using full bincount, change this
        

        # Flatten initial qubit matrix to store in dataframe
        df_qubit = init_toric.qubit_matrix.reshape((-1))

        # Create indices for generated data
        names = ['data_nr', 'layer', 'x', 'y']
        index_qubit = pd.MultiIndex.from_product([[i], np.arange(2),
                                                 np.arange(params['size']),
                                                 np.arange(params['size'])],
                                                 names=names)
        index_distr = pd.MultiIndex.from_product([[i], np.arange(16)+2, [0],
                                                 [0]], names=names)

        # Add data to Dataframes
        df_qubit = pd.DataFrame(df_qubit.astype(np.uint8), index=index_qubit,
                                columns=['data'])
        df_distr = pd.DataFrame(df_eq_distr.astype(np.uint8),  # dtype for eq_distr? want uint16
                                index=index_distr, columns=['data'])

        # Add dataframes to temporary list to shorten computation time
        
        df_list.append(df_qubit)
        df_list.append(df_distr)

        # Every x iteration adds data to data file from temporary list
        # and clears temporary list
        
        if (i + 1) % 3 == 0:
            df = df.append(df_list)
            df_list.clear()
            print('Intermediate save point reached (writing over)')
            df.to_pickle(file_path)

    # Adds any remaining data from temporary list to data file when run is over
    if len(df_list) > 0:
        df = df.append(df_list)
        print('\nSaving all generated data (writing over)')
        df.to_pickle(file_path)

    print('\nCompleted')


if __name__ == '__main__':
    # All paramteters for data generation is set here,
    # some of which may be irrelevant depending on the choice of others
    t_start = time.time()
    params = {'size': 5,
              'p': 0.185,
              'Nc': 9,
              'steps': 500000,
              'iters': 10,
              'conv_criteria': 'error_based',
              'SEQ': 7,
              'TOPS': 10,
              'eps': 0.005}

    # Get job array id, set working directory, set timer
    try:
        array_id = str(sys.argv[1])
        local_dir = str(sys.argv[2])
        timeout = int(sys.argv[3])
    except:
        array_id = '0'
        local_dir = '.'
        timeout = 100000000000
        print('invalid sysargs')

    # Build file path
    file_path = os.path.join(local_dir, 'data_' + array_id + '.xz')

    # Generate data
    generate(file_path, params, timeout, method="ST")

    # View data file
    
    iterator = MCMCDataReader(file_path, params['size'])
    while iterator.has_next():
        print('Datapoint nr: ' + str(iterator.current_index() + 1))
        print(iterator.next())
