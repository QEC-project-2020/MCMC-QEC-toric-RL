import copy  # not used
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns



from datetime import datetime

from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.mcmc import *
from decoders import STDC, STRC, single_temp
from generate_data import pickle_reader


combine_two = True
#size = int(sys.argv[1])
#p = float(sys.argv[2])
#steps = int(sys.argv[3])
#filepaths = [sys.argv[i] for i in range(1,len(sys.argv))]
filepaths = []

for root, dirs, files in os.walk("./complexity_data/tmp", topdown=False):
   for name in files:
      #print(os.path.join(root, name))
      filepaths.append(name)

"""params = {'size': size,
          'p': p,
          'Nc': 9,
          'steps': steps,
          'iters': 10,
          'eps': 0.005,
          'p_sampling': 0.25}"""

#methods = ["PTEQ", "ST", "STDC", "STRC", "STDC_rain"]
methods = ["STDC","ST", "PTEQ"]

size_array = np.zeros(len(filepaths))
final_success_rate = np.zeros(len(filepaths))





merge = True
frames = []
if merge ==True:
    for k in range(len(filepaths)):
        print(k)
        frames.append(pickle_reader("./complexity_data/tmp/" + filepaths[k]))
    result = pd.concat(frames, axis=0, join='outer', ignore_index=False, keys=None,levels=None, names=None, verify_integrity=False, copy=True)

for k in range(1):
    file = filepaths[0]
    raindrops = 1
    print(file, "here")
    unpickled_df  = result #pickle_reader("complexity_data/" + file)
    qubits = result['data'].to_numpy()
    parse = file.split('_')
    size = int(parse[1])
    steps = int(parse[2])
    p = float(parse[3])
    p_sampling = 0.25

    print(size, "ready")

    for method in methods:
        eq_steps = unpickled_df["eq_steps_" + method].tolist()
        num_points = len(eq_steps[0])
        print("Number of datapoints: ", num_points)
        init_code = Planar_code(size)
        success = np.zeros((len(qubits),num_points-1))
        #print('180', len(qubits),len(data[0]))
        success_rate = np.zeros((num_points-1))
        print(eq_steps[-1][-1]),

        for i in range(len(qubits)):
            #print("------")
            #print(i)
            #print(len(qubits), "Len qubits")
            #print("data", i, data[i])
            for j in range(num_points-1):
                #if i == 136:
                #    print(eq_steps[i][:])
                #print(len(eq_steps[i]), method)
                if  len(eq_steps[i]) >= num_points-1:
                    #print(method, "here1", len(eq_steps[i]))
                    q = np.asarray(qubits[i])
                    d = np.asarray(eq_steps[i][j])
                    init_code.qubit_matrix = q
                    a =  init_code.define_equivalence_class()
                else:
                    #print(method, "here2")
                    success[i, j] = 1
                    continue
                # NB: if ST use argmin
                if method == "STDC" or method == "STRC" or method == "PTEQ" or method == "STDC_rain" or method == "STRC_rain" or method=="STDC_rain_fast":
                    if a == np.argmax(d):
                        success[i, j] = 1

                elif method == "ST":
                    if a == np.argmin(d):
                        success[i, j] = 1
        if method == "PTEQ":
            success_PTEQ = success
        if method == "STDC":
            success_STDC = success


        for j in range(len(success_rate)):
            success_rate[j] = np.sum(success[:, j])/(len(success[:, j]))
            print(np.sum(success[:, j]), (len(success[:, j])))
        #print(success_rate, 'success_rate', method)
        if method == "STDC_rain" or method == "STRC_rain" or method == "STRC_rain_fast":
            x = np.linspace(0, int(steps) , len(success_rate))
        else: x = np.linspace(0, int(steps) , len(success_rate))


        plt.figure(1)
        line = plt.plot(x, success_rate, label = method)
        alpha = 0.05
        z = 1
        standard_error = z*np.sqrt(success_rate*(1-success_rate)/len(success[:, 0])) #np.std(success, axis=0)
        plt.fill_between(x, success_rate - standard_error, success_rate + standard_error, interpolate=True, alpha=0.2)
        #plt.errorbar(x, success_rate, yerr=np.std(success_rate,axis=0), xerr=None)
        #line = plt.plot(1/x*size**8, np.log((1-success_rate)*size**2), label = str(size))
        #line = plt.plot(1/x*np.exp(0.001*size), np.log((1-success_rate)*size**2), label = str(size))

        """plt.figure(1)
        alpha = (4-2.2)/(2.9-1.6)
        beta = 6
        c = np.exp(-2.5+1.8*alpha)
        print("c",c,"alpha",alpha)
        line = plt.plot(1/x*size**beta, np.log((1-success_rate)/c), label = str(size))
        final_success_rate[k] = success_rate[-1]
        size_array[k] = size


        plt.figure(3)
        line = plt.plot(x/size**beta, (1-success_rate), label = str(size))"""

        #line = plt.plot(1/x[:-1]*size**10, np.diff(np.log((1-success_rate)*size**2))/np.diff(1/x*size**6) , label = str(size)) #derivative
        #line = plt.plot(1/x, np.log((1-success_rate)*size**1.5), label = str(size))
        #print(len(eq_steps[0]), "eq steps")

print(len(eq_steps[0]))
print("num steps", steps)
plt.legend()

PATH = './steps_graphs/'
now = datetime.now()
timestamp = str(datetime.timestamp(now))
print("timestamp =", timestamp)

filename = './steps_graphs/' + str(steps) + '_' + str(p) + '_' + str(size) + '_' + timestamp + '.png'
"""plt.figure(1)
plt.savefig(filename)
plt.show()"""

"""plt.figure(2)
line = plt.plot(np.log(size_array), np.log(1-final_success_rate))
#plt.savefig(filename)
plt.show()"""

plt.figure(1)
#plt.xscale('symlog')
plt.savefig(filename)
plt.show()

plt.figure(2)
success_rate_correlated = np.zeros(len(success_rate))
for j in range(len(success_rate)):
    success_rate_correlated[j] = (np.sum(success_STDC[:, j]*success_PTEQ[:, j])/(len(success_STDC[:, j])))/(np.sum(success_PTEQ[:, j])/(len(success_PTEQ[:, j])))
#plt.xscale('symlog')
plt.plot(x, success_rate_correlated)
plt.show()

for j in range(len(success_STDC[:, -1])):
    a = success_STDC[j, -1]*success_PTEQ[j, -1]
    if a > 0:
        print(qubits[j])
        print("----------")
