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
from decoders import *
#from error_rate_history import *


# This function generates training data with help of the MCMC algorithm
#profile

def generate(file_path, params, timeout, max_capacity=10**4, nbr_datapoints=10**6, method="PTEC"):

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

    #counts number of logical errors for each algorithm
    logical_error_counter_PTEQ = 0
    logical_error_counter_ST = 0
    logical_error_counter_STDC = 0
    logical_error_counter_STRC = 0
    logical_error_counter_STDC_rain = 0
    logical_error_counter_STRC_rain = 0

    logical_error_limit = params['logical_error_limit']

    #needed for numbda compilation


    # Loop to generate data points


    #fix to stop numba from compiling every iteration
    #init_code = Planar_code(params['size'])
    #df_eq_distr = single_temp(init_code, params['p'], 100 , mwpm_start = params['mwpm_start'])
    #df_eq_distr = STDC_rain_fast(init_code, init_code.system_size, params['p_sampling'], droplets = params['raindrops'], steps = 100,  mwpm_start =  params['mwpm_start'])
    #df_eq_distr = STDC(init_code, size = params['size'], p_error = params['p'], p_sampling = params['p_sampling'], steps=100, mwpm_start = params['mwpm_start'])





    for i in np.arange(nbr_to_generate) + nbr_existing_data:
        # Breaks if run has exceeded timeout-value
        if time.time() - t_start > timeout:
            print("timeout reached: " + str(timeout) + "s")
            break

        print('Starting generation of point nr: ' + str(i + 1))

        # Initiate toric
        init_code = Planar_code(params['size'])
        init_code.generate_random_error(params['p'])
        start_code = copy.deepcopy(init_code)
        df_qubit = start_code.qubit_matrix.tolist()



        if params['mwpm_start'] == False:
            init_code.qubit_matrix, _ = init_code.apply_random_logical()
            init_code.qubit_matrix = init_code.apply_stabilizers_uniform() # fix so all uses these


        df_eq_distr = single_temp(init_code, params['p'], steps = params['steps'], mwpm_start = params['mwpm_start'])
        df_eq_distr7 = np.array(df_eq_distr)


        #df_eq_distr = STDC_rain_fast(init_code, init_code.system_size, params['p'], p_sampling = params['p_sampling'], droplets = params['raindrops'], steps = int(params['steps']/params['raindrops']),  mwpm_start =  params['mwpm_start'])
        #df_eq_distr6 = np.array(df_eq_distr)
        df_eq_distr = STDC(init_code, init_code.system_size, params['p'], p_sampling = params['p_sampling'], steps = params['steps'],  mwpm_start =  params['mwpm_start'])
        df_eq_distr6 = np.array(df_eq_distr)


        df_eq_distr = PTEQ(init_code, params['p'], steps=params['steps'], Nc = params['Nc'], tops_burn = params['tops_burn'], mwpm_start = params['mwpm_start'])
        df_eq_distr5 = np.array(df_eq_distr)

        if np.argmax(df_eq_distr6[:,-1]) != start_code.define_equivalence_class():
            logical_error_counter_STDC+=1
        print("failures STDC:", logical_error_counter_STDC)
        if np.argmax(df_eq_distr5[:,-1]) != start_code.define_equivalence_class():
            logical_error_counter_PTEQ +=1
        print("failures PTEQ:", logical_error_counter_PTEQ)
        if np.argmin(df_eq_distr7[:,-1]) != start_code.define_equivalence_class():
            logical_error_counter_ST +=1
        print("failures ST:", logical_error_counter_ST)
        print("Test", flush=True)


        names = ['data_nr', 'layer', 'x', 'y']



        df_eq_distr5 = (df_eq_distr5.transpose()).tolist()
        df_eq_distr6 = (df_eq_distr6.transpose()).tolist()
        df_eq_distr7 = (df_eq_distr7.transpose()).tolist()

        #df_entry = pd.DataFrame([[df_qubit, df_eq_distr0, df_eq_distr1, df_eq_distr2, df_eq_distr3, df_eq_distr4]], columns = ['data', 'eq_steps_PTEQ' , 'eq_steps_ST', 'eq_steps_STDC','eq_steps_STRC', 'eq_steps_STDC_rain']) #['data'])
        df_entry = pd.DataFrame([[df_qubit, df_eq_distr6, df_eq_distr5, df_eq_distr7]], columns = ['data','eq_steps_STDC', 'eq_steps_PTEQ' ,'eq_steps_ST']) #['data'])
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

def main():
    # All paramteters for data generation is set here,
    # some of which may be irrelevant depending on the choice of others
    t_start = time.time()
    nbr_datapoints = 1

    mwpm_start = True

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
              'p': 0.13,
              'Nc': None,    #OBS! Should be equal to system size d
              'steps': 1000,#int(5*(int(array_id)**5)/1000)*1000, #int(5*(int(array_id)**5)/1000)*1000, #Needs to divide number of data poins
              'iters': 10,
              'conv_criteria': 'error_based',
              'SEQ': 15,
              'TOPS': 10,
              'eps': 0.005,
              'tops_burn': 0,
              'p_sampling': 0.25,
              'raindrops': 1,
              'mwpm_start': mwpm_start,
              'logical_error_limit': 2000,
              'nbr_eq_classes': 4}

    now = datetime.now()
    timestamp = str(datetime.timestamp(now))
    print("timestamp =", timestamp)

    file_path = os.path.join(local_dir, 'data_' + array_id + '_' + str(params['steps']) + '_' + str(params['p']) + '_' + str(params['size']) + '_' + 'avg' + str(nbr_datapoints) + '_psample' + str(params['p_sampling'])+ '_' + str(params['raindrops']) + '_'+  timestamp + '.xz')


    # Generate data
    #methods = ["PTEQ", "ST", "STDC", "STRC", "STDC_rain"]
    methods = ["STDC", "PTEQ", "ST"]
    generate(file_path, params, timeout, nbr_datapoints = nbr_datapoints)
    unpickled_df  = pickle_reader(file_path)
    qubits = unpickled_df['data'].to_numpy()

    for method in methods:
        eq_steps = unpickled_df["eq_steps_" + method].tolist()
        init_code = Planar_code(params['size'])
        success = np.zeros((len(qubits),len(eq_steps[0])))
        mwpm_success = np.zeros((len(qubits)))
        success_rate = np.zeros(len(eq_steps[0]))

        for i in range(len(qubits)):
            #print("data", i, data[i])
            for j in range(len(eq_steps[0])):

                q = np.asarray(qubits[i])
                d = np.asarray(eq_steps[i][j])


                init_code.qubit_matrix = q
                a =  init_code.define_equivalence_class()
                # NB: if ST use argmin
                if method == "STDC" or method == "STRC" or method == "PTEQ" or method == "STDC_rain" or method == "STRC_rain" or method == "STDC_rain_fast":
                    if a == np.argmax(d):
                        success[i, j] = 1


                elif method == "ST":
                    if a == np.argmin(d):
                        success[i, j] = 1

        for j in range(len(success_rate)):
            success_rate[j] = np.sum(success[:, j])/(len(success[:, j]))
        print(success_rate, 'success_rate', method)

        #mwpm_success_rate = np.sum(mwpm_success)/len(mwpm_success)

        if method == "STRC_rain" or method == "STDC_rain"or method == "STDC_rain_fast":
            x = np.linspace(0, int(params['steps']), len(success_rate))
        else:
            x = np.linspace(0, int(params['steps']) , len(success_rate))
        line = plt.plot(x, success_rate, label = method)
        plt.hlines(np.amax(success_rate), 0, int(params['steps']), linestyles='dotted')
        #plt.hlines(mwpm_success_rate, 0, int(params['steps']), linestyles='dashed')

    print("num steps", params['steps'])
    plt.xlim(0, params['steps'])
    plt.legend()

    PATH = './steps_graphs/'
    now = datetime.now()
    timestamp = str(datetime.timestamp(now))
    print("timestamp =", timestamp)

    filename = './steps_graphs/' + 'size_' + array_id + '_' + str(params['steps']) + '_' + str(params['p']) + '_' + str(params['size']) + '_' + 'avg' + str(nbr_datapoints) + '_psample' + str(params['p_sampling'])+ '_' + str(params['raindrops']) + '_'+  timestamp+ '.png'
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    main()















    # View data file
    #iterator = MCMCDataReader(file_path, params['size'])

    #while iterator.has_next():
    #    print('Datapoint nr: ' + str(iterator.current_index() + 1))
    #    print(iterator.next())
