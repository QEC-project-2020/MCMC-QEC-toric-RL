from src.toric_model import Toric_code
from src.planar_model import Planar_code, _define_equivalence_class
from src.mcmc import *
from math import *
from decoders import *
import numpy as np
import copy
import pandas as pd
import time
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

def plot_Nobs():
    # size = 9
    # p = 0.1
    # beta = -np.log((p / 3) / (1 - p))
    # init_code = Planar_code(size)
    # init_code.generate_random_error(p)
    # init_code = class_sorted_mwpm(init_code)
    # Nobs_n = STDC_Nall_n(init_code, p, p, steps=5*size**5)

    # fig = plt.figure()
    # ax = fig.add_subplot()
    
    # for i in range(4):
    #     lists = sorted(Nobs_n[i].items()) # sorted by key, return a list of tuples

    #     x, y = zip(*lists) # unpack a list of pairs into two tuples

    #     minlen = min(x)

    #     x = np.array(x[:10]) - minlen
    #     y = np.array(y[:10])

    #     ax.plot(x, np.log(y*np.exp(beta*x)), '*--')
    
    # fig.savefig(f'plots/test.pdf', bbox_inches='tight')



    fig = plt.figure(figsize=(8,12))
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    X_d = []
    Y_d = [[],[],[]]

    for size_nr, size in enumerate([5, 7, 9, 13, 15, 19, 25, 35, 41, 45]):   
        #print(f'\nSize {size}')
        for p_nr, p_err in enumerate([0.05]):
            #print(f'P_err {p_err}')
            beta = -np.log((p_err / 3) / (1 - p_err))

            avg_num = 50

            maxlen = 60

            X = np.arange(maxlen)
            Y = np.zeros((maxlen))

            for i in range(avg_num):
                #print('Syndrom nr', i + 1, 'of', avg_num)
                init_code = Planar_code(size)
                init_code.generate_random_error(p_err)
                init_code.apply_stabilizers_uniform()
                init_code = class_sorted_mwpm(init_code)
                # burn-in

                # for d = 45, s ~ 20000
                Nobs_n = STDC_Nall_n(init_code, p_err, p_err, steps=20000)
                Nobs_n = STDC_Nall_n(init_code, p_err, p_err, steps=2000000)

                ## check for burn-in?

                # fig2 = plt.figure()
                # axx1 = fig2.add_subplot(2,1,1)
                # axx2 = fig2.add_subplot(2,1,2)
                # for i in range(4):
                #     axx1.plot(acf(np.array(path[i]), 100000))
                    
                #     axx2.plot(np.array(path[i]))

                # plt.savefig('plots/test_burnin.pdf')



                for j in range(4):
                    lists = sorted(Nobs_n[j].items()) # sorted by key, return a list of tuples

                    x, y = zip(*lists) # unpack a list of pairs into two tuples

                    minlen = min(x)

                    x = np.array(x[:maxlen]) - minlen
                    y = np.array(y[:maxlen])

                    for length in range(x.shape[0]):
                        Y[length] += y[length]
                    
            
            filled_len = np.argmax(np.nonzero(Y))

            Y_data = np.log((Y/Y[0])*np.exp(beta*X))

            
            print('attempting save')
            np.save('testX.np', X)
            np.save('testY.np', Y_data)

            lo = int(filled_len/3*2)
            hi = filled_len

            print('derivative between', lo, hi)
            fit = np.polyfit(X[lo:hi], Y_data[lo:hi], 1)
            
            ax1.plot(X, Y_data, label=f'd={size}')
            ax1.axvline(8)
            ax1.axvline(14)
            ax1.axvline(6)
            ax1.grid(True)
            ax1.set_xlabel('Chain length (0 at shortest)')
            ax1.legend()

            ax2.plot(X, (Y), label=f'd={size}')
            ax2.grid(True)
            ax2.set_xlabel('Chain length (0 at shortest)')
            ax2.legend()

            X_d.append(size)
            Y_d[0].append(fit[0])
            Y_d[1].append((np.log((Y/Y[0])*np.exp(beta*X))[14] - np.log((Y/Y[0])*np.exp(beta*X))[8])/6)
            Y_d[2].append(np.log((Y/Y[0])*np.exp(beta*X))[6]/6)

            ax3.clear()
            ax3.plot(X_d, Y_d[0], label=f'linfit')
            ax3.plot(X_d, Y_d[1], label=f'derivate 8-14')
            ax3.plot(X_d, Y_d[2], label=f'offset={6}')
            
            s = np.linspace(1, 23)
            ax3.plot(s, np.log(s), 'k')
            ax3.grid(True)
            ax3.set_xlabel('d')
            ax3.legend()
            print(X_d)
            print(Y_d[1])

            if size_nr >=2:
                ax4.clear()
                ax4.plot(np.log(X_d), Y_d[2], '*-', label=f'derivate 0-6')
                ax4.plot(np.log(X_d), Y_d[0], '*-', label=f'linfit')
                ax4.set_xlabel('log(d)')
                ax4.set_ylabel('')
                ax4.grid(True)
                ax4.legend()
                print(np.polyfit(np.log(X_d), Y_d[0], 1))
                #ax.set_ylabel('Chain length (0 at shortest)')
        
            plt.savefig(f'plots/N_n_p005.pdf', bbox_inches='tight')


if __name__ == '__main__':

    plot_Nobs()
