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
from datetime import datetime


# This function generates training data with help of the MCMC algorithm
def generate(file_path, params, max_capacity=10**5, nbr_datapoints=10**6,
             fixed_errors=None):

    if params['code'] == 'planar':
        nbr_eq_class = 4
    elif params['code'] == 'toric':
        nbr_eq_class = 16

    if params['method'] == "all":
        nbr_eq_class *= 3

    # Creates df
    df = pd.DataFrame()
    nbr_existing_data = 0

    # Add parameters to dataframe
    names = ['data_nr', 'type']
    index_params = pd.MultiIndex.from_product([[-1], np.arange(1)],
                                                names=names)
    df_params = pd.DataFrame([[params]],
                            index=index_params,
                            columns=['data'])
    df = df.append(df_params)
                                
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
        elif params['code'] == 'planar':
            init_code = Planar_code(params['size'])

        ################### DO NOT SPLIT ######################################
        # Generate random error                                            ####
        init_code.generate_random_error(params['p_error'])                 ####
                                                                           ####
        # Flatten initial qubit matrix to store in dataframe               ####
        df_qubit = copy.deepcopy(init_code.qubit_matrix)                   ####
        eq_true = init_code.define_equivalence_class()                     ####
        ################### DO NOT SPLIT ######################################

        # Start in completely random state, does not change syndrom
        init_code.qubit_matrix, _ = init_code.apply_random_logical()
        init_code.qubit_matrix = init_code.apply_stabilizers_uniform()

        # Save this object for later usage in pteq if mwpm init is True
        init_code_pre_mwpm = copy.deepcopy(init_code)

        if params['mwpm_init']:  # get mwpm starting points
            init_code = class_sorted_mwpm(init_code)
            print('Starting in MWPM state')
        else:
            print('Starting in random state')

        # Generate data for DataFrame storage  OBS now using full bincount,
        # change this
        if params['method'] == "PTEQ":
            # If we have an emwpm list, find the best class and use that as
            # init code. Note that we prefer lower class numbers
            best_class = 0
            shortest = 9999999999
            if type(init_code) == list:
                for j in range(len(init_code)):
                    if init_code[j].count_errors() < shortest:
                        shortest = init_code[j].count_errors()
                        best_class = j

            # Run PTEQ with the shortest class
            df_eq_distr, n_steps = PTEQ(init_code[best_class],
                                        params['p_error'])
            # Add number of steps so that total can be calculated later
            df_eq_distr = np.concatenate((df_eq_distr, np.array([n_steps])),
                                            axis=0)
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        if params['method'] == "PTDC":
            df_eq_distr, conv = PTDC(init_code,
                                     params['p_error'],
                                     params['p_sampling'])
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        if params['method'] == "PTRC":
            df_eq_distr, conv = PTRC(init_code,
                                     params['p_error'],
                                     params['p_sampling'])
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "STDC":
            df_eq_distr = STDC(init_code,
                               params['p_error'],
                               params['p_sampling'],
                               steps=params['steps'],
                               droplets=params['droplets'])
            df_eq_distr = np.array(df_eq_distr)
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "STDC_N_n":
            df_eq_distr = STDC_Nall_n(init_code,
                               params['p_error'],
                               params['p_sampling'],
                               steps=params['steps'])
            df_eq_distr = np.array(df_eq_distr)
        elif params['method'] == "ST":
            df_eq_distr = single_temp(init_code,
                                      params['p_error'],
                                      params['steps'])
            df_eq_distr = np.array(df_eq_distr)
            if np.argmin(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "STRC":
            df_eq_distr = STRC(init_code,
                               params['p_error'],
                               p_sampling=params['p_sampling'],
                               steps=params['steps'],
                               droplets=params['droplets'])
            df_eq_distr = np.array(df_eq_distr)
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "all":
            # init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr1 = single_temp(init_code,
                                       params['p_error'],
                                       params['steps'])

            # init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr2 = STDC(init_code,
                                params['p_error'],
                                p_sampling=params['p_sampling'],
                                steps=params['steps'],
                                droplets=params['droplets'])

            # init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr3 = STRC(init_code,
                                params['p_error'],
                                p_sampling=params['p_sampling'],
                                steps=params['steps'],
                                droplets=params['droplets'])

            df_eq_distr = np.concatenate((df_eq_distr1,
                                          df_eq_distr2,
                                          df_eq_distr3), axis=0)
        elif params['method'] == "STDC_PTEQ_stepcomp":

            pteq_converted_steps = int(params['steps'] / (params['Nc'] * params['iters']))
            stdc_converted_steps = int(params['steps'] / (nbr_eq_class * 5))

            # calculate using PTEQ
            df_eq_distr1, n_step = PTEQ(init_code_pre_mwpm,
                               params['p_error'],
                               steps=pteq_converted_steps)

            # calculate using STDC
            df_eq_distr2 = STDC(init_code,
                                params['p_error'],
                                p_sampling=params['p_sampling'],
                                steps=stdc_converted_steps,
                                droplets=params['droplets'])

            df_eq_distr = np.concatenate((df_eq_distr1,
                                          df_eq_distr2), axis=0)
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

        # Generate data for DataFrame storage  OBS now using full
        # bincount, change this

        # Create indices for generated data
        names = ['data_nr', 'type']
        index_qubit = pd.MultiIndex.from_product([[i], np.arange(1)],
                                                 names=names)
        index_distr = pd.MultiIndex.from_product([[i], np.arange(1)+1],
                                                 names=names)

        # Add data to Dataframes
        df_qubit = pd.DataFrame([[df_qubit.astype(np.uint8)]],
                                index=index_qubit,
                                columns=['data'])
        df_distr = pd.DataFrame([[df_eq_distr]],
                                index=index_distr,
                                columns=['data'])

        # Add dataframes to temporary list to shorten computation time

        df_list.append(df_qubit)
        df_list.append(df_distr)

        # Every x iteration adds data to data file from temporary list
        # and clears temporary list

        # this contant needs to be sufficiently big that rsync has time
        # to sync files before update, maybe change this to be
        # time-based instead.
        if (i + 1) % 100 == 0:
            df = df.append(df_list)
            df_list.clear()
            print('Intermediate save point reached (writing over)')
            df.to_pickle(file_path)
            print('Failed so far:', failed_syndroms)

        # If the desired amount of errors have been achieved, break the
        # loop and finish up
        if failed_syndroms == fixed_errors:
            print('Desired amount of failes syndroms achieved, breaking loop.')
            break

    # Adds any remaining data from temporary list to data file when run
    # is over
    if len(df_list) > 0:
        df = df.append(df_list)
        print('\nSaving all generated data (writing over)')
        df.to_pickle(file_path)

    print('\nCompleted')


if __name__ == '__main__':
    # Get job array id, working directory
    job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    array_id = os.getenv('SLURM_ARRAY_TASK_ID')
    local_dir = os.getenv('TMPDIR')
    size = int(3 + 2 * int(int(array_id) / 160 + 0.0001) + 0.0001)
    print('size:', size)
    params = {'code':           "planar",
              'method':         "STDC_N_n",
              'size':           size,
              'p_error':        np.round(int(int(array_id)%16)*0.01+ 0.05, decimals=2),#np.round(int(array_id/50)*0.005+ 0.17, decimals=3),#np.round((0.05 + float(int(array_id) % 32) / 200), decimals=3),
              'p_sampling':     0.25,
              'droplets':       1,
              'mwpm_init':      True,
              'fixed_errors':   None,
              'Nc':             size,
              'iters':          10,
              'conv_criteria':  'None',
              'SEQ':            2,
              'TOPS':           10,
              'eps':            0.1}
    # Steps is a function of code size L
    params.update({'steps': int(5 * params['size'] ** 5)})

    print('Nbr of steps to take if applicable:', params['steps'])

    # Build file path
    file_path = os.path.join(local_dir, 'data_id_' + job_id + '_' + array_id + '_size_' + str(size) + '_STDC_results.xz')

    # Generate data
    generate(file_path, params, nbr_datapoints=2, fixed_errors=params['fixed_errors'])

    # View data file
    '''iterator = MCMCDataReader(file_path, params['size'])
    data = iterator.full()
    params = data[0]
    data = np.delete(data, 0)
    print(params)
    for k in np.arange(int(len(data)/2)):
        qubit_matrix = data[2*k].reshape(2,params['size'],params['size'])
        eq_distr = data[2*k+1]

        print(qubit_matrix)
        print(eq_distr)'''
