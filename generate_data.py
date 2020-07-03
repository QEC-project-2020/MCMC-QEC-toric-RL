import copy  # not used
import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.mcmc import *
from decoders import STDC, STRC, single_temp


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
        init_code = Planar_code(params['size'])
        init_code.generate_random_error(params['p'])

        #randomize input matrix, no trace of seed.
        input_matrix, _ = apply_random_logical(init_code.qubit_matrix)
        input_matrix = init_code.apply_stabilizers_uniform() # fix so all uses these
        init_code.qubit_matrix = input_matrix

        # Generate data for DataFrame storage  OBS now using full bincount, change this
        if method == "PTEC":
            df_eq_distr, _, _ = parallel_tempering(init_code, params['Nc'],
                                         p=params['p'], steps=params['steps'],
                                         iters=params['iters'],
                                         conv_criteria=params['conv_criteria'])

        df_eq_distr= single_temp(init_code, params['p'], params['steps'])
        df_eq_distr1 = np.array(df_eq_distr)

        df_eq_distr = STDC(init_code, size = params['size'], p_error = params['p'], p_sampling = params['p'], steps=params['steps'])
        df_eq_distr2 = np.array(df_eq_distr)

        df_eq_distr = STRC(init_code, size = init_code.system_size, p_error = params['p'], p_sampling= params['p'], steps=params['steps'])
        df_eq_distr3 = np.array(df_eq_distr)


        #else:
        #    raise ValueError('Invalid method, use "PTEC", "STDC" or "ST".')

        # Generate data for DataFrame storage  OBS now using full bincount, change this

        #print((df_eq_distr.transpose()).tolist(), 'df_eq_distr')

        #df_eq_distr = #(df_eq_distr.transpose()).tolist()

        # Flatten initial qubit matrix to store in dataframe
        df_qubit = init_code.qubit_matrix.tolist()

        # Create indices for generated data
        names = ['data_nr', 'layer', 'x', 'y']

        """index_qubit = pd.MultiIndex.from_product([[i], np.arange(2),
                                                 np.arange(params['size']),
                                                 np.arange(params['size'])],
                                                 names=names)
        index_distr = pd.MultiIndex.from_product([[i], [2], [0],
                                                 [0]], names=names)"""


        # Add data to Dataframes
        #df_qubit = pd.DataFrame(df_qubit.astype(np.uint8), index=index_qubit,
                                #columns=['data'])
        df_eq_distr1 = (df_eq_distr1.transpose()).tolist() #  = df_eq_distr.reshape((-1))
        df_eq_distr2 = (df_eq_distr2.transpose()).tolist()
        df_eq_distr3 = (df_eq_distr3.transpose()).tolist()
        #steps = i*params['steps']

        df_entry = pd.DataFrame([[df_qubit, df_eq_distr1, df_eq_distr2, df_eq_distr3]], columns = ['data', 'eq_steps_ST', 'eq_steps_STDC','eq_steps_STRC']) #['data'])

        # Add dataframes to temporary list to shorten computation time

        #df_list.append(df_qubit)
        df_list.append(df_entry)

        # Every x iteration adds data to data file from temporary list
        # and clears temporary list

        if (i + 1) % 100 == 0:
            df = df.append(df_list, ignore_index = True)
            df_list.clear()
            print('Intermediate save point reached (writing over)')
            df.to_pickle(file_path)

    # Adds any remaining data from temporary list to data file when run is over
    if len(df_list) > 0:
        df = df.append(df_list, ignore_index = True)
        print('\nSaving all generated data (writing over)')
        df.to_pickle(file_path)

    #print(df)
    print('\nCompleted')

def pickle_reader(file_path):
    unpickled_df = pd.read_pickle(str(file_path))
    return unpickled_df




if __name__ == '__main__':
    # All paramteters for data generation is set here,
    # some of which may be irrelevant depending on the choice of others
    t_start = time.time()
    nbr_datapoints = 500



    method="STDC"

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
    params = {'size': int(array_id),
              'p': 0.05,
              'Nc': 9,
              'steps': int((10000 * (int(array_id)/5)**4)/100)*100,
              'iters': 10,
              'conv_criteria': 'error_based',
              'SEQ': 7,
              'TOPS': 10,
              'eps': 0.005,
              'p_sampling': 0.05}
    now = datetime.now()
    timestamp = str(datetime.timestamp(now))
    print("timestamp =", timestamp)

    file_path = os.path.join(local_dir, 'data_' + array_id + '_' + str(params['steps']) + '_' + str(params['p']) + '_' + str(params['size']) + timestamp + '.xz')


    # Generate data
    methods = ["ST", "STDC", "STRC"]
    generate(file_path, params, timeout, method=method, nbr_datapoints = nbr_datapoints)
    unpickled_df  = pickle_reader(file_path)
    qubits = unpickled_df['data'].to_numpy()


    for method in methods:
        eq_steps = unpickled_df["eq_steps_" + method].tolist()
        init_code = Planar_code(params['size'])
        success = np.zeros((len(qubits),len(eq_steps[0])))
        #print('180', len(qubits),len(data[0]))
        success_rate = np.zeros(len(eq_steps[0]))
        for i in range(len(qubits)):
            #print("data", i, data[i])
            for j in range(len(eq_steps[0])):

                q = np.asarray(qubits[i])
                d = np.asarray(eq_steps[i][j])
                init_code.qubit_matrix = q
                a =  init_code.define_equivalence_class()
                # NB: if ST use argmin
                if method == "STDC" or method == "STRC":
                    if a == np.argmax(d):
                        success[i, j] = 1
                elif method == "ST":
                    if a == np.argmin(d):
                        success[i, j] = 1

        for j in range(len(success_rate)):
            success_rate[j] = np.sum(success[:, j])/(len(success[:, j]))
        print(success_rate, 'success_rate', method)
        line = plt.plot(success_rate, label = method)

    print("num steps", params['steps'])
    plt.legend()

    PATH = './steps_graphs/'
    now = datetime.now()
    timestamp = str(datetime.timestamp(now))
    print("timestamp =", timestamp)

    filename = './steps_graphs/' + str(params['steps']) + '_' + str(params['p']) + '_' + str(params['size']) + '_' + timestamp + '.png'
    plt.savefig(filename)
    #plt.show()















    # View data file
    #iterator = MCMCDataReader(file_path, params['size'])

    #while iterator.has_next():
    #    print('Datapoint nr: ' + str(iterator.current_index() + 1))
    #    print(iterator.next())
