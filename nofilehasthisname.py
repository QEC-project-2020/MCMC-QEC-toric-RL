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
import seaborn as sns
from labellines import labelLine, labelLines
import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from operator import itemgetter

#sns.set_style("whitegrid")

from matplotlib import rc
#rc('font',**{'family':'sans-serif'})#,'sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif'})#,'serif':['Palatino']})
rc('text', usetex=True)


def runstdc():
    size = 5
    init_code = Planar_code(size)
    p_error = 0.20
    init_code.generate_random_error(p_error)
    init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
    print(init_code.qubit_matrix)
    print('Correct eq-class = ', init_code.define_equivalence_class(), '( coulumn', init_code.define_equivalence_class()+1,')')
    for i in range(10):
        eq = STDC(init_code, size, p_error = p_error, p_sampling=p_error, steps = 10000)
        print('Try nr.', i+1, ':\t',eq)

def getMCMCstats(qubit_matrix_in, size, p_error, Nc=19, steps=1000000, crit='error_based'):
    """Get statistics about distribution of error chains given a syndrom (error chain) using MCMC-sampling."""
    init_toric = Toric_code(size)
    
    # define error
    init_toric.qubit_matrix = qubit_matrix_in
    init_toric.syndrom('next_state')

    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')
    # Start in random eq-class
    init_toric.qubit_matrix, _ = apply_random_logical(init_toric.qubit_matrix)

    distr, count, qubitlist = parallel_tempering_plus(init_toric, Nc, p=p_error, steps=steps, iters=10, conv_criteria=crit)
    print(distr)
    unique_elements, unique_counts = np.unique(qubitlist, axis=0, return_counts=True)
    print('Number of unique elements: ', len(unique_elements))

    shortest = inf
    for i in range(len(qubitlist)):
        nb = np.count_nonzero(qubitlist[i])
        if nb < shortest:
            shortest = nb

    # save all qubits to df_all
    df = pd.DataFrame({"qubit":[], "nbr_err":[], "nbr_occ":[], "eq_class":[]})
    df = pd.concat((pd.DataFrame({"qubit":[unique_elements[i]], "nbr_err":[np.count_nonzero(unique_elements[i])], "nbr_occ":[unique_counts[i]], "eq_class": define_equivalence_class(unique_elements[i])}) for i in range(len(unique_elements))),
            ignore_index=True)
    
    for i in range(7):
        print(shortest+i)
        print(df.loc[df['nbr_err'] == shortest + i])
        for j in range(16):
            nbr_comb = len(df.loc[df['nbr_err'] == shortest + i].loc[df['eq_class'] == j])
            if nbr_comb > 0:
                print('class ', j, '\t\t', nbr_comb)

def N_n_3d():
    size = 5
    init_code = Planar_code(size)
    

    ### LOOK AT SYNDROM
    init_code.qubit_matrix = np.array([[[1, 0, 0, 0, 0],
                                        [3, 0, 3, 0, 0],
                                        [0, 0, 0, 0, 1],
                                        [0, 0, 0, 3, 0],
                                        [0, 0, 0, 0, 0]],
                                       [[0, 3, 0, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 3, 0, 3, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0]]], dtype=np.uint8)

    _, qubitlist = STDC_biased(init_code, np.array([0.10, 0.10, 0.10]), 0.25, steps=5*7**5)
    
    fig = plt.figure()
    ax = Axes3D(fig)

    markers = ['.', '1', 'v', '*']
    markers = ['.', '.', '.', '.']
    
    
    for i in range(4):
        eq = qubitlist[i]
        
        # nx, ny, nz, antal funna
        d = {}

        for j, key in enumerate(eq):
            hashable = tuple(eq[key])
            if hashable in d:
                d[hashable] += 1
            else:
                d[hashable] = 1

        points = np.zeros((len(d), 4))

        smallest = 100

        for j, key in enumerate(d):
            s = key[0] + key[1] + key[2]
            if s < smallest:
                smallest = s
            points[j][0] = key[0]
            points[j][1] = key[1] 
            points[j][2] = key[2]
            points[j][3] = d[key]
        
        print(points)
        
        def f(x, y, a):
            r = a - x - y
            mask = r < 0
            r[mask] = 0
            return r
        
        

       

        # surface for the model

        

        p_error = 0.1
        beta = -np.log((p_error / 3) / (1 - p_error))


        # vikta med antal också så att storlek beror direkt på hur mycket de bidrar till summan givet brusmodell
        #ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=points[:,3]*3000*np.exp(-beta*(points[:, 0]+ points[:, 1]+ points[:, 2]))/np.exp(-beta*7), zorder = 10, marker=markers[i])
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10*points[:,3], zorder = 10, marker=markers[i])
    x = y = np.arange(0, 7, 1)
    X, Y = np.meshgrid(x, y)
    zs = np.array(f(np.ravel(X), np.ravel(Y), 7.5))
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, alpha=0.2)
    zs = np.array(f(np.ravel(X), np.ravel(Y), 10.5))
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, alpha=0.2)
    zs = np.array(f(np.ravel(X), np.ravel(Y), 9.5))
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, alpha=0.2)
    zs = np.array(f(np.ravel(X), np.ravel(Y), 8.5))
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, alpha=0.2)
    
    ax.plot([0, 6], [0, 0], [0, 0], zdir='z', color='k', linewidth=5)
    ax.plot([0, 0], [0, 6], [0, 0], zdir='x', color='k', linewidth=5)
    ax.plot([0, 0], [0, 0], [0, 6], zdir='y', color='k', linewidth=5)
    ax.set_xlabel('Number of x errors')
    ax.set_ylabel('Number of y errors')
    ax.set_zlabel('Number of z errors')
    ax.set_xlim((0,6))
    ax.set_ylim((0,6))
    ax.set_zlim((0,6))
    plt.tight_layout()
    plt.show()

    # fig = plt.figure()
    # for i in range(4):
    #     ax = fig.add_subplot(2,2,i+1)    
    #     N_n = []

    #     for key in qubitlist[i]:
    #         N_n.append(qubitlist[i][key])
        
    #     ax.hist(N_n, bins=np.unique(N_n)) # make this a bar-plot
    #     plt.xlim((5,15))
    #     plt.title(f'{i}')
    #     plt.yscale('log')
    
    # plt.savefig('plots/test.pdf')

def N_n():
    fig = plt.figure(figsize=(8,10))
    for size_nr, (size, xlims) in tqdm(enumerate([(5, (3, 15)), (7, (5, 20)), (9, (8, 25))])):   
        #print(f'\nSize {size}')
        for p_nr, p_err in tqdm(enumerate([0.05, 0.2])):
            #print(f'P_err {p_err}')

            # ### LOOK AT SYNDROM
            # init_code.qubit_matrix = np.array([[[1, 0, 0, 0, 0],
            #                                     [3, 0, 3, 0, 0],
            #                                     [0, 0, 0, 0, 1],
            #                                     [0, 0, 0, 3, 0],
            #                                     [0, 0, 0, 0, 0]],
            #                                    [[0, 3, 0, 0, 0],
            #                                     [0, 0, 0, 0, 0],
            #                                     [0, 3, 0, 3, 0],
            #                                     [0, 0, 0, 1, 0],
            #                                     [0, 0, 0, 0, 0]]], dtype=np.uint8)
            # init_code.qubit_matrix = np.array([[[0, 0, 0, 0, 0, 0, 0],
            #                                     [0, 0, 0, 0, 0, 0, 0],
            #                                     [0, 0, 0, 0, 1, 0, 3],
            #                                     [0, 0, 0, 0, 0, 0, 0],
            #                                     [0, 0, 0, 0, 0, 3, 2],
            #                                     [0, 0, 2, 2, 0, 3, 0],
            #                                     [0, 0, 0, 0, 2, 0, 0]],
            #                                    [[0, 0, 0, 0, 0, 0, 0],
            #                                     [0, 0, 2, 0, 0, 0, 0],
            #                                     [0, 0, 0, 0, 0, 3, 0],
            #                                     [1, 0, 1, 0, 0, 0, 0],
            #                                     [0, 0, 2, 0, 0, 0, 0],
            #                                     [0, 0, 0, 1, 0, 0, 0],
            #                                     [0, 0, 0, 0, 0, 0, 0]]], dtype=np.uint8)
            avg_num = 10

            qubitlist_lop = [{}, {}, {}, {}]
            qubitlist_hip = [{}, {}, {}, {}]
            for i in tqdm(range(avg_num)):
                #print('Syndrom nr', i + 1, 'of', avg_num)
                init_code = Planar_code(size)
                init_code.generate_random_error(p_err)
                init_code = class_sorted_mwpm(init_code)
                _, new_lop = STDC_returnall(init_code, p_err, p_err, steps=5*9**5, droplets=1)
                _, new_hip = STDC_returnall(init_code, p_err, 0.25, steps=5*9**5, droplets=1)
                
                for j in range(4):
                    _, key_min_lop = min(new_lop[j].items(), key=itemgetter(1))
                    new_lop[j] = {key: new_lop[j][key] - key_min_lop for key in new_lop[j]}
                    qubitlist_lop[j].update(new_lop[j])
                    _, key_min_hip = min(new_hip[j].items(), key=itemgetter(1))
                    new_hip[j] = {key: new_hip[j][key] - key_min_hip for key in new_hip[j]}
                    qubitlist_hip[j].update(new_hip[j])
            
            ax = fig.add_subplot(3, 2, 2*size_nr + p_nr + 1)#3,2, size_nr + 1)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            N_n_all_lop = []
            N_n_all_hip = []
            for i in range(4):
                for key in qubitlist_lop[i]:
                    N_n_all_lop.append(qubitlist_lop[i][key])
                for key in qubitlist_hip[i]:
                    N_n_all_hip.append(qubitlist_hip[i][key])

            #test1, counts1 = np.unique(np.array(N_n_all), return_counts=True)
            #ax.plot(test1, np.log(counts1/avg_num), 'k', lw=5, alpha = 0.7)

            for p_sample, qubitlist in [(p_err, qubitlist_lop), (0.25, qubitlist_hip)]:
                #ax = fig.add_subplot(2,2,i+1)    
                N_n = []

                for i in range(4):
                    for key in qubitlist[i]:
                        N_n.append(qubitlist[i][key])

                test, counts = np.unique(np.array(N_n), return_counts=True)
                
                #ax.hist(N_n, bins=np.unique(N_n)) # make this a bar-plot
                ax.plot(test, np.log(counts/avg_num/4), '-*', label=f'psample {p_sample}')
                #ax.set_xlim(0, xlims[1])
                #ax.set_yticks([1,2,3,4,5,6,7,8,9,10])
                
                #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.grid(True)
                ax.legend()
                plt.title(f'L: {size} perror: {p_err}')
                #plt.yscale('log')
        
            plt.savefig(f'plots/N_n_avg10_all_eq.pdf', bbox_inches='tight')

def N_n_expbetan():
    fig = plt.figure(figsize=(8,20))
    for size_nr, (size, xlims) in tqdm(enumerate([(5, (3, 15)), (7, (5, 20)), (9, (8, 25)), (11, (3, 15)), (13, (5, 20)), (15, (8, 25))])):   
        #print(f'\nSize {size}')
        for p_nr, p_err in tqdm(enumerate([0.05, 0.2])):
            #print(f'P_err {p_err}')

            avg_num = 10

            qubitlist_lop = [{}, {}, {}, {}]
            qubitlist_hip = [{}, {}, {}, {}]
            for i in tqdm(range(avg_num)):
                #print('Syndrom nr', i + 1, 'of', avg_num)
                init_code = Planar_code(size)
                init_code.generate_random_error(p_err)
                init_code = class_sorted_mwpm(init_code)
                _, new_lop = STDC_returnall(init_code, p_err, p_err, steps=5*size**5, droplets=1)
                _, new_hip = STDC_returnall(init_code, p_err, 0.25, steps=5*size**5, droplets=1)
                
                for j in range(4):
                    temp = {}
                    temp.update(new_lop[j])
                    temp.update(new_hip[j])
                    _, spo = min(temp.items(), key=itemgetter(1))
                    new_lop[j] = {key: new_lop[j][key] - key_min for key in new_lop[j]}
                    qubitlist_lop[j].update(new_lop[j])
                    #_, key_min_hip = min(new_hip[j].items(), key=itemgetter(1))
                    new_hip[j] = {key: new_hip[j][key] - key_min for key in new_hip[j]}
                    qubitlist_hip[j].update(new_hip[j])
            
            ax = fig.add_subplot(6, 2, 2*size_nr + p_nr + 1)#3,2, size_nr + 1)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            N_n_all_lop = []
            N_n_all_hip = []
            for i in range(4):
                for key in qubitlist_lop[i]:
                    N_n_all_lop.append(qubitlist_lop[i][key])
                for key in qubitlist_hip[i]:
                    N_n_all_hip.append(qubitlist_hip[i][key])

            #test1, counts1 = np.unique(np.array(N_n_all), return_counts=True)
            #ax.plot(test1, np.log(counts1/avg_num), 'k', lw=5, alpha = 0.7)

            for p_sample, qubitlist in [(p_err, qubitlist_lop), (0.25, qubitlist_hip)]:## still need to fix plotting ax x=0
                #ax = fig.add_subplot(2,2,i+1)    
                N_n = []

                for i in range(4):
                    for key in qubitlist[i]:
                        N_n.append(qubitlist[i][key])

                test, counts = np.unique(np.array(N_n), return_counts=True)
                
                #ax.hist(N_n, bins=np.unique(N_n)) # make this a bar-plot

                beta = -np.log((p_err / 3) / (1 - p_err))

                ax.plot(test, (np.exp(-beta*np.arange(0,counts.shape[0]))*counts/avg_num/4), '-*', label=f'psample {p_sample}')
                #ax.set_xlim(0, xlims[1])
                #ax.set_yticks([1,2,3,4,5,6,7,8,9,10])
                
                #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.grid(True)
                ax.legend()
                plt.title(f'L: {size} perror: {p_err}')
                #plt.yscale('log')
        
            plt.savefig(f'plots/N_n_betan.pdf', bbox_inches='tight')

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def acf(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])

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

            for i in tqdm(range(avg_num)):
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

def plot_p():
    size = 7
    init_code = Planar_code(size)
    

    ### LOOK AT SYNDROM
    init_code.qubit_matrix = np.array([[[0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 3],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 3, 2],
                                        [0, 0, 2, 2, 0, 3, 0],
                                        [0, 0, 0, 0, 2, 0, 0]],
                                       [[0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 2, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 3, 0],
                                        [1, 0, 1, 0, 0, 0, 0],
                                        [0, 0, 2, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0]]], dtype=np.uint8)

    init_code.plot('planar_eqflip_7x7')
    #init_code = class_sorted_mwpm(init_code)


    p = (0.01 + 0.01*i for i in range(20))

    l = np.zeros((20, 4))
    for i, p_err in enumerate(p):
        #print(p_err)
        distr = STDC(init_code, p_err, p_sampling=0.25, droplets=1, steps=5*7**5)
        #print(PTEQ(init_code, 0.10))
        l[i,:] = distr
        print(distr)

    p = (0.01 + 0.01*i for i in range(20))
    p_np = np.array(list(p))

    fig, ax = plt.subplots(figsize=(4,3))

    for i in range(4):
        ax.plot(p_np, l[:,i], label=f'eq {i}')
    ax.legend()
    ax.set_xlabel(r'$p_{error}$')
    ax.set_ylabel(r'$\mathrm{P}(p_{error}, \mathrm{eq})$')
    plt.savefig('plots/planar_eqflip_7x7_eq.pdf', bbox_inches='tight')
    ### FIND SYNDROM
    # for i in tqdm(range(10000)):
    #     init_code = Planar_code(size)
    #     init_code.generate_random_error(0.15)
    #     pre = copy.deepcopy(init_code)
    #     init_code = class_sorted_mwpm(init_code)
    #     d1 = STDC(init_code, 0.01, p_sampling=0.25, droplets=1, steps=5**5)
    #     d2 = STDC(init_code, 0.10, p_sampling=0.25, droplets=1, steps=5**5)
    #     best1 = np.argmax(d1)
    #     best2 = np.argmax(d2)
    #     if best1 != best2:
    #         print(pre.qubit_matrix)
    #         print(d1, d2)
    #         print(STDC(init_code, 0.01, p_sampling=0.25, droplets=1, steps=5*7**5), STDC(init_code, 0.10, p_sampling=0.25, droplets=1, steps=5*7**5))

    

def create_ALL_df_for_figure_7():
    file_prefix = 'data/all/'

    nbr_sizes = 1
    start_size = 25
    step_size = 2

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02

    correct_guesses_ST = np.zeros((nbr_sizes,nbr_p))
    correct_guesses_STDC = np.zeros((nbr_sizes,nbr_p))
    correct_guesses_STRC = np.zeros((nbr_sizes,nbr_p))

    total_guesses = np.zeros((nbr_sizes,nbr_p))
    
    P_ST = np.zeros((nbr_sizes,nbr_p))
    P_STDC = np.zeros((nbr_sizes,nbr_p))
    P_STRC = np.zeros((nbr_sizes,nbr_p))

    x = [np.round(start_p + step_p*i, decimals=2) for i in range(nbr_p)] 

    #std = np.zeros((3,16))


    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(nbr_sizes):
        for j in range(nbr_p):
            size = np.round(start_size + i * step_size, decimals=2)
            print()
            print('Processing...\nSize:', size)
            p = np.round(start_p + j * step_p, decimals=2)
            print('p_error:', p)

            file_suffix = 'data_size_' + str(size) + '_method_all_id_' + str(j) + '_perror_' + str(p) + 'special10drop1e4.xz'
            file_path = file_prefix + file_suffix
            print(file_path)

            iterator = MCMCDataReader(file_path, size)
            while iterator.has_next():
                qubit_matrix, eq_distr = iterator.next()
                eq_ST = eq_distr[0:4]
                eq_STDC = eq_distr[4:8]
                eq_STRC = eq_distr[8:12]

                true_eq = _define_equivalence_class(qubit_matrix)
                predicted_eq_ST = np.argmin(eq_ST)
                predicted_eq_STDC = np.argmax(eq_STDC)
                predicted_eq_STRC = np.argmax(eq_STRC)

                if predicted_eq_ST == true_eq:
                    correct_guesses_ST[i,j] += 1
                if predicted_eq_STDC == true_eq:
                    correct_guesses_STDC[i,j] += 1
                if predicted_eq_STRC == true_eq:
                    correct_guesses_STRC[i,j] += 1
                total_guesses[i,j] += 1
            P_ST[i,j] = correct_guesses_ST[i,j]/total_guesses[i,j]
            #std[i,j] = sqrt(total_guesses[i,j]*(1-P_ST[i,j])*P_ST[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
            #df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P_ST[i,j], "nbr_pts":total_guesses[i,j], "method":'ST'}))

            P_STDC[i,j] = correct_guesses_STDC[i,j]/total_guesses[i,j]
            #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
            df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P_STDC[i,j], "nbr_pts":total_guesses[i,j], "method":'STDC'}))

            P_STRC[i,j] = correct_guesses_STRC[i,j]/total_guesses[i,j]
            #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
            df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P_STRC[i,j], "nbr_pts":total_guesses[i,j], "method":'STRC'}))
    
    df.to_pickle('data/fig7_ALL_data_spec.xz')

def create_ALL_df_for_figure_7_psample():
    file_prefix = 'data/all/'

    nbr_sizes = 6
    start_size = 0.16
    step_size = 0.03

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02

    correct_guesses_ST = np.zeros((nbr_sizes,nbr_p))
    correct_guesses_STDC = np.zeros((nbr_sizes,nbr_p))
    correct_guesses_STRC = np.zeros((nbr_sizes,nbr_p))

    total_guesses = np.zeros((nbr_sizes,nbr_p))
    
    P_ST = np.zeros((nbr_sizes,nbr_p))
    P_STDC = np.zeros((nbr_sizes,nbr_p))
    P_STRC = np.zeros((nbr_sizes,nbr_p))

    x = [np.round(start_p + step_p*i, decimals=2) for i in range(nbr_p)] 

    #std = np.zeros((3,16))


    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(nbr_sizes):
        for j in range(nbr_p):
            size = np.round(start_size + i * step_size, decimals=2)
            print()
            print('Processing...\nSize:', size)
            p = np.round(start_p + j * step_p, decimals=2)
            print('p_error:', p)

            file_suffix = 'data_size_' + str(11) + '_method_all_id_' + str(j) + '_perror_' + str(p) + '_psample_'+ str(size)+ '.xz'
            file_path = file_prefix + file_suffix
            print(file_path)
            #try:
            iterator = MCMCDataReader(file_path, 11)
            while iterator.has_next():
                qubit_matrix, eq_distr = iterator.next()
                eq_ST = eq_distr[0:4]
                eq_STDC = eq_distr[4:8]
                eq_STRC = eq_distr[8:12]

                true_eq = _define_equivalence_class(qubit_matrix)
                predicted_eq_ST = np.argmin(eq_ST)
                predicted_eq_STDC = np.argmax(eq_STDC)
                predicted_eq_STRC = np.argmax(eq_STRC)

                if predicted_eq_ST == true_eq:
                    correct_guesses_ST[i,j] += 1
                if predicted_eq_STDC == true_eq:
                    correct_guesses_STDC[i,j] += 1
                if predicted_eq_STRC == true_eq:
                    correct_guesses_STRC[i,j] += 1
                total_guesses[i,j] += 1
            P_ST[i,j] = correct_guesses_ST[i,j]/total_guesses[i,j]
            #std[i,j] = sqrt(total_guesses[i,j]*(1-P_ST[i,j])*P_ST[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
            #df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P_ST[i,j], "nbr_pts":total_guesses[i,j], "method":'ST'}))

            P_STDC[i,j] = correct_guesses_STDC[i,j]/total_guesses[i,j]
            #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
            df = df.append(pd.DataFrame({"p":[p], "psamp": str(size), "P":P_STDC[i,j], "nbr_pts":total_guesses[i,j], "method":'STDC'}))

            P_STRC[i,j] = correct_guesses_STRC[i,j]/total_guesses[i,j]
            #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
            df = df.append(pd.DataFrame({"p":[p], "psamp": str(size), "P":P_STRC[i,j], "nbr_pts":total_guesses[i,j], "method":'STRC'}))
            #except:
            #    print('failed!')
            #    continue
    df.to_pickle('data/fig7_ALL_data_11x11_psamp.xz')

def create_ALL_df_for_figure_7_ndrop():
    file_prefix = 'data/all/'

    nbr_sizes = 6
    start_size = 0.16
    step_size = 0.03

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02

    droplets = [1, 2, 5, 10]

    correct_guesses_ST = np.zeros((nbr_sizes,nbr_p))
    correct_guesses_STDC = np.zeros((nbr_sizes,nbr_p))
    correct_guesses_STRC = np.zeros((nbr_sizes,nbr_p))

    total_guesses = np.zeros((nbr_sizes,nbr_p))
    
    P_ST = np.zeros((nbr_sizes,nbr_p))
    P_STDC = np.zeros((nbr_sizes,nbr_p))
    P_STRC = np.zeros((nbr_sizes,nbr_p))

    x = [np.round(start_p + step_p*i, decimals=2) for i in range(nbr_p)] 

    #std = np.zeros((3,16))


    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(4):
        ndrop = droplets[i]
        for j in range(nbr_p):
            size = np.round(start_size + i * step_size, decimals=2)
            print()
            print('Processing 15x15...\nNdrop:', ndrop)
            p = np.round(start_p + j * step_p, decimals=2)
            print('p_error:', p)

            file_suffix = 'data_size_' + str(15) + '_method_all_id_' + str(j) + '_perror_' + str(p) + '_ndrop_'+ str(ndrop)+ '.xz'
            file_path = file_prefix + file_suffix
            print(file_path)
            try:
                iterator = MCMCDataReader(file_path, 15)
                while iterator.has_next():
                    qubit_matrix, eq_distr = iterator.next()
                    eq_ST = eq_distr[0:4]
                    eq_STDC = eq_distr[4:8]
                    eq_STRC = eq_distr[8:12]

                    true_eq = _define_equivalence_class(qubit_matrix)
                    predicted_eq_ST = np.argmin(eq_ST)
                    predicted_eq_STDC = np.argmax(eq_STDC)
                    predicted_eq_STRC = np.argmax(eq_STRC)

                    if predicted_eq_ST == true_eq:
                        correct_guesses_ST[i,j] += 1
                    if predicted_eq_STDC == true_eq:
                        correct_guesses_STDC[i,j] += 1
                    if predicted_eq_STRC == true_eq:
                        correct_guesses_STRC[i,j] += 1
                    total_guesses[i,j] += 1
                P_ST[i,j] = correct_guesses_ST[i,j]/total_guesses[i,j]
                #std[i,j] = sqrt(total_guesses[i,j]*(1-P_ST[i,j])*P_ST[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
                #df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P_ST[i,j], "nbr_pts":total_guesses[i,j], "method":'ST'}))

                P_STDC[i,j] = correct_guesses_STDC[i,j]/total_guesses[i,j]
                #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
                df = df.append(pd.DataFrame({"p":[p], "ndrop": str(ndrop), "P":P_STDC[i,j], "nbr_pts":total_guesses[i,j], "method":'STDC'}))

                P_STRC[i,j] = correct_guesses_STRC[i,j]/total_guesses[i,j]
                #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
                df = df.append(pd.DataFrame({"p":[p], "ndrop": str(ndrop), "P":P_STRC[i,j], "nbr_pts":total_guesses[i,j], "method":'STRC'}))
            except:
                print('failed!')
                continue
    df.to_pickle('data/fig7_ALL_data_15x15_ndrop.xz')

def create_df_for_figure_7():
    file_prefix = 'data/new/'

    sizes = [5, 7, 9, 11, 13, 15, 17]

    nbr_sizes = len(sizes)

    nbr_p = 32
    start_p = 0.05
    step_p = 0.005

    correct_guesses = np.zeros((nbr_sizes,nbr_p))

    total_guesses = np.zeros((nbr_sizes,nbr_p))
    
    P = np.zeros((nbr_sizes,nbr_p))


    x = [np.round(start_p + step_p*i, decimals=3) for i in range(nbr_p)] 

    df = pd.DataFrame()

    for i in range(len(sizes)):
        size = sizes[i]
        for j in range(nbr_p):
            print()
            print('Processing...\nSize:', size)
            p = np.round(start_p + j * step_p, decimals=3)
            print('p_error:', p)

            file_suffix = 'data_size_' + str(size) + '_method_STDC_id_' + str(i*nbr_p + j) + '_perror_' + str(p) + '_better-resL4.xz'
            file_path = file_prefix + file_suffix
            print(file_path)

            iterator = MCMCDataReader(file_path, size)

            data = iterator.full()
            for k in range(int(len(data)/2)):
                qubit_matrix = data[2*k].reshape(2,size,size)
                eq_distr = data[2*k+1]

                #print(qubit_matrix)
                #print(eq_distr)

                true_eq = _define_equivalence_class(qubit_matrix)
                predicted_eq = np.argmax(eq_distr)

                if predicted_eq == true_eq:
                    correct_guesses[i,j] += 1
                total_guesses[i,j] += 1

            P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
            #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
            df = df.append(pd.DataFrame({"p":[p], "d": str(size)+ 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "method":'STDCL4'}))
    
    print(df)
    df.to_pickle('data/fig7_STDC_5*(5)*L4_data_hi_res.xz')

def create_extra():
    file_prefix = 'data/new/'

    sizes = [17, 19, 21, 23, 25]

    nbr_sizes = len(sizes)

    nbr_p = 5
    start_p = 0.17
    step_p = 0.005

    correct_guesses = np.zeros((nbr_sizes,nbr_p))

    total_guesses = np.zeros((nbr_sizes,nbr_p))
    
    P = np.zeros((nbr_sizes,nbr_p))


    x = [np.round(start_p + step_p*i, decimals=3) for i in range(nbr_p)] 

    df = pd.DataFrame()

    for i in range(len(sizes)):
        size = sizes[i]
        for j in range(nbr_p):
            print()
            print('Processing...\nSize:', size)
            p = np.round(start_p + j * step_p, decimals=3)
            print('p_error:', p)

            # Loop over all paralell processes
            for pp in range(10): # pp for paralell process
                file_suffix = 'data_size_' + str(size) + '_method_STDC_id_' + str(10*(i + j*nbr_sizes)+pp) + '_perror_' + str(p) + '_extra.xz'
                file_path = file_prefix + file_suffix
                print(file_path)

                iterator = MCMCDataReader(file_path, size)

                data = iterator.full()
                for k in range(int(len(data)/2)):
                    qubit_matrix = data[2*k].reshape(2,size,size)
                    eq_distr = data[2*k+1]

                    #print(qubit_matrix)
                    #print(eq_distr)

                    true_eq = _define_equivalence_class(qubit_matrix)
                    predicted_eq = np.argmax(eq_distr)

                    if predicted_eq == true_eq:
                        correct_guesses[i,j] += 1
                    total_guesses[i,j] += 1

            P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
            #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
            df = df.append(pd.DataFrame({"p":[p], "d": str(size)+ 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "method":'STDCL5'}))
    
    print(df)
    df.to_pickle('data/fig7_STDC_5*(5)*L5_data_hi_res_extra.xz')


def create_fromnewstructure():
    file_prefix = 'data/new/'

    # sort twice to get files in natural order (first sorted job-id then array-id as ints)
    file_path_list = sorted(sorted(glob.glob(file_prefix + '*extra2size*'), key=lambda a: int(a.split("_")[3])), key=lambda a: int(a.split("_")[2]))

    print(f'Processing {len(file_path_list)} files.')

    result_matrix = {}

    for file_path in file_path_list:
        # Load file
        df = pd.read_pickle(file_path) # 91% of the time, no point optimizing
        data = df.to_numpy().ravel()

        # Extract paramters
        params = data[0]
        # Delete paramter data from data
        data = np.delete(data, 0)
        
        print('Processing...', file_path)

        p_error = params['p_error']
        size = params['size']

        for k in range(int(len(data)/2)):
            qubit_matrix = data[2*k].reshape(2,size,size)
            eq_distr = data[2*k+1]

            true_eq = _define_equivalence_class(qubit_matrix)
            predicted_eq = np.argmax(eq_distr)

            if size not in result_matrix:
                result_matrix[size] = {}
            if p_error not in result_matrix[size]:
                result_matrix[size][p_error] = {'corr_guess':0, 'total_guess':0}

            result_matrix[size][p_error]['total_guess'] += 1

            if predicted_eq == true_eq:
                result_matrix[size][p_error]['corr_guess'] += 1

    df = pd.DataFrame()

    for size in result_matrix:
        for p_error in result_matrix[size]:
            P = result_matrix[size][p_error]['corr_guess'] / result_matrix[size][p_error]['total_guess']
            result_matrix[size][p_error]['P'] = P
            print(size, p_error, result_matrix[size][p_error])
            df = df.append(pd.DataFrame({"p":[p_error], "d": str(size)+ 'x' + str(size), "P":P, "nbr_pts":result_matrix[size][p_error]['total_guess'], "method":'STDCL5'}))

    print(df)
    #df.to_pickle('data/fig7_STDC_5*(5)*L5_data_hi_res_extra2.xz')

def create_fromnewstructure_comp():
    file_prefix = 'data/new/'
    job_id = '1900876'

    # sort twice to get files in natural order (first sorted job-id then array-id as ints)
    file_path_list = sorted(sorted(glob.glob(file_prefix + '*' + job_id + '*'), key=lambda a: int(a.split("_")[3])), key=lambda a: int(a.split("_")[2]))

    print(f'Processing {len(file_path_list)} files.')

    results = {}

    n_steps = []

    for file_path in file_path_list:

        # Load file
        df = pd.read_pickle(file_path) # 91% of the time, no point optimizing
        data = df.to_numpy().ravel()

        # Extract paramters
        params = data[0]
        param_H = hash(frozenset(params.items()))

        # Delete paramter data from data
        data = np.delete(data, 0)
        
        print('Processing...', file_path)

        p_error = params['p_error']
        size = params['size']

        for k in range(int(len(data)/2)):
            qubit_matrix = data[2*k].reshape(2,size,size)
            eq_distr = data[2*k+1]
            
            true_eq = _define_equivalence_class(qubit_matrix)
            predicted_eq_pteq = np.argmax(eq_distr[:4])

            #n_steps.append(eq_distr[4])
            
            #print(qubit_matrix, eq_distr, true_eq, predicted_eq_pteq, predicted_eq_stdc)
            #print(true_eq, predicted_eq_pteq)

            #print(true_eq, predicted_eq_pteq, predicted_eq_stdc, ' \n \n')

            if param_H not in results:
                results[param_H] = {'corr_guess_stdc':0, 'corr_guess_pteq':0, 'total_guess':0, 'size':params['size'], 'p_error':params['p_error'], 'method':params['method']}

            results[param_H]['total_guess'] += 1

            if predicted_eq_pteq == true_eq:
                results[param_H]['corr_guess_pteq'] += 1

    #print(np.mean(n_steps))
    df = pd.DataFrame()

    # Add to dataframe once alla data for the same settings have been added
    for key in results.keys():
        p_error = results[key]['p_error']
        size = results[key]['size']
        P = results[key]['corr_guess_pteq'] / results[key]['total_guess']

        #print(size, p_error, P_stdc, P_pteq)
        df = df.append(pd.DataFrame({"p":[p_error], "d": str(size)+ 'x' + str(size), "P":P, "nbr_pts":results[key]['total_guess'], "method":results[key]['method']}))

    print(df)
    df.to_pickle('data/fig7_' + job_id + '.xz')

def create_eMWPM_df_for_figure_7():
    file_prefix = 'data/new/'

    nbr_sizes = 1
    start_size = 5
    step_size = 2

    sizes = [5, 7, 9, 11, 13, 15, 17]

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02

    correct_guesses = np.zeros((len(sizes),nbr_p))

    total_guesses = np.zeros((len(sizes),nbr_p))
    
    P = np.zeros((len(sizes),nbr_p))


    x = [np.round(start_p + step_p*i, decimals=2) for i in range(nbr_p)] 

    #std = np.zeros((3,16))
    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(len(sizes)):
        size = sizes[i]
        for j in range(nbr_p):
            try:
                print()
                print('Processing...\nSize:', size)
                p = np.round(start_p + j * step_p, decimals=2)
                print('p_error:', p)

                file_suffix = 'data_size_' + str(size) + '_method_eMWPM_id_' + str(i*nbr_p + j) + '_perror_' + str(p) + '_fig7.xz'
                file_path = file_prefix + file_suffix
                print(file_path)

                iterator = MCMCDataReader(file_path, size)

                data = iterator.full()
                for k in range(int(len(data)/2)):
                    qubit_matrix = data[2*k].reshape(2,size,size)
                    eq_distr = data[2*k+1]

                    #print(qubit_matrix)
                    #print(eq_distr)

                    true_eq = _define_equivalence_class(qubit_matrix)
                    predicted_eq = np.argmax(eq_distr)

                    if predicted_eq == true_eq:
                        correct_guesses[i,j] += 1
                    total_guesses[i,j] += 1

                P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
                #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
                #df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "method":'STDCr'}))
                #df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "nbr_failed": (total_guesses[i,j] - correct_guesses[i,j]), "method":'eMWPM'}), ignore_index=True)
                df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "method":'eMWPM'}), ignore_index=True)
            except:
                print('failed, moving on')
                continue
    
    print(df)
    df.to_pickle('data/fig7_eMWPM_15data_fig7.xz')

def create_PTDC_df_for_figure_7():
    file_prefix = 'data/new/'

    nbr_sizes = 1
    start_size = 5
    step_size = 2

    sizes = [15]

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02

    correct_guesses = np.zeros((len(sizes),nbr_p))

    total_guesses = np.zeros((len(sizes),nbr_p))
    
    P = np.zeros((len(sizes),nbr_p))


    x = [np.round(start_p + step_p*i, decimals=2) for i in range(nbr_p)] 

    #std = np.zeros((3,16))
    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(len(sizes)):
        size = sizes[i]
        for j in range(nbr_p):
            try:
                print()
                print('Processing...\nSize:', size)
                p = np.round(start_p + j * step_p, decimals=2)
                print('p_error:', p)

                file_suffix = 'data_size_' + str(size) + '_method_PTDC_id_' + str(j) + '_perror_' + str(p) + '_2000err.xz'
                file_path = file_prefix + file_suffix
                print(file_path)

                iterator = MCMCDataReader(file_path, size)

                data = iterator.full()
                for k in range(int(len(data)/2)):
                    qubit_matrix = data[2*k].reshape(2,size,size)
                    eq_distr = data[2*k+1]

                    #print(qubit_matrix)
                    #print(eq_distr)

                    true_eq = _define_equivalence_class(qubit_matrix)
                    predicted_eq = np.argmax(eq_distr)

                    if predicted_eq == true_eq:
                        correct_guesses[i,j] += 1
                    total_guesses[i,j] += 1

                P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
                #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
                df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "nbr_failed": (total_guesses[i,j] - correct_guesses[i,j]), "method":'PTDC'}), ignore_index=True)
            except:
                print('failed, moving on')
                continue
    
    print(df)
    df.to_pickle('data/fig7_PTDC_15data_2000.xz')

def create_MWPM_df_for_figure_7():
    file_prefix = 'data/new/'

    nbr_sizes = 1
    start_size = 5
    step_size = 2

    sizes = [15]

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02

    correct_guesses = np.zeros((len(sizes),nbr_p))

    total_guesses = np.zeros((len(sizes),nbr_p))
    
    P = np.zeros((len(sizes),nbr_p))


    x = [np.round(start_p + step_p*i, decimals=2) for i in range(nbr_p)] 

    #std = np.zeros((3,16))
    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(len(sizes)):
        size = sizes[i]
        for j in range(nbr_p):
            try:
                print()
                print('Processing...\nSize:', size)
                p = np.round(start_p + j * step_p, decimals=2)
                print('p_error:', p)

                file_suffix = 'data_size_' + str(size) + '_method_MWPM_id_' + str(j) + '_perror_' + str(p) + '2000err.xz'
                file_path = file_prefix + file_suffix
                print(file_path)

                iterator = MCMCDataReader(file_path, size)

                data = iterator.full()
                for k in range(int(len(data)/2)):
                    qubit_matrix = data[2*k].reshape(2,size,size)
                    eq_distr = data[2*k+1]

                    #print(qubit_matrix)
                    #print(eq_distr)

                    true_eq = _define_equivalence_class(qubit_matrix)
                    predicted_eq = np.argmax(eq_distr)

                    if predicted_eq == true_eq:
                        correct_guesses[i,j] += 1
                    total_guesses[i,j] += 1

                P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
                #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
                df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "nbr_failed": (total_guesses[i,j] - correct_guesses[i,j]), "method":'MWPM'}), ignore_index=True)
            except:
                print('failed, moving on')
                continue
    
    print(df)
    df.to_pickle('data/fig7_MWPM_15data_2000.xz')

def create_STDC_2000err():
    file_prefix = 'data/new/'

    nbr_sizes = 1
    start_size = 5
    step_size = 2

    sizes = [15]

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02

    correct_guesses = np.zeros((len(sizes),nbr_p))

    total_guesses = np.zeros((len(sizes),nbr_p))
    
    P = np.zeros((len(sizes),nbr_p))


    x = [np.round(start_p + step_p*i, decimals=2) for i in range(nbr_p)] 

    #std = np.zeros((3,16))
    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(len(sizes)):
        size = sizes[i]
        for j in range(nbr_p):
            try:
                print()
                print('Processing...\nSize:', size)
                p = np.round(start_p + j * step_p, decimals=2)
                print('p_error:', p)

                file_suffix = 'data_size_' + str(size) + '_method_STDC_id_' + str(j) + '_perror_' + str(p) + '_steps_50625_2000err.xz'
                file_path = file_prefix + file_suffix
                print(file_path)

                iterator = MCMCDataReader(file_path, size)

                data = iterator.full()
                for k in range(int(len(data)/2)):
                    qubit_matrix = data[2*k].reshape(2,size,size)
                    eq_distr = data[2*k+1]

                    #print(qubit_matrix)
                    #print(eq_distr)

                    true_eq = _define_equivalence_class(qubit_matrix)
                    predicted_eq = np.argmax(eq_distr)

                    if predicted_eq == true_eq:
                        correct_guesses[i,j] += 1
                    total_guesses[i,j] += 1

                P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
                #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
                df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "nbr_failed": (total_guesses[i,j] - correct_guesses[i,j]), "method":'STDCL4'}), ignore_index=True)
            except:
                print('failed, moving on')
                continue
    
    print(df)
    df.to_pickle('data/fig7_STDC_15data_2000_L4.xz')

def create_ST_2000err():
    file_prefix = 'data/new/'

    nbr_sizes = 1
    start_size = 5
    step_size = 2

    sizes = [15]

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02

    correct_guesses = np.zeros((len(sizes),nbr_p))

    total_guesses = np.zeros((len(sizes),nbr_p))
    
    P = np.zeros((len(sizes),nbr_p))


    x = [np.round(start_p + step_p*i, decimals=2) for i in range(nbr_p)] 

    #std = np.zeros((3,16))
    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(len(sizes)):
        size = sizes[i]
        for j in range(nbr_p):
            try:
                print()
                print('Processing...\nSize:', size)
                p = np.round(start_p + j * step_p, decimals=2)
                print('p_error:', p)

                file_suffix = 'data_size_' + str(size) + '_method_ST_id_' + str(j) + '_perror_' + str(p) + '_steps_50625_2000err.xz'
                file_path = file_prefix + file_suffix
                print(file_path)

                iterator = MCMCDataReader(file_path, size)

                data = iterator.full()
                for k in range(int(len(data)/2)):
                    qubit_matrix = data[2*k].reshape(2,size,size)
                    eq_distr = data[2*k+1]

                    #print(qubit_matrix)
                    #print(eq_distr)

                    true_eq = _define_equivalence_class(qubit_matrix)
                    predicted_eq = np.argmin(eq_distr)

                    if predicted_eq == true_eq:
                        correct_guesses[i,j] += 1
                    total_guesses[i,j] += 1

                P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
                #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
                df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "nbr_failed": (total_guesses[i,j] - correct_guesses[i,j]), "method":'STL4'}), ignore_index=True)
            except:
                print('failed, moving on')
                continue
    
    print(df)
    df.to_pickle('data/fig7_ST_15data_2000_L4.xz')

def create_STRC_2000err():
    file_prefix = 'data/new/'

    nbr_sizes = 1
    start_size = 5
    step_size = 2

    sizes = [15]

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02

    correct_guesses = np.zeros((len(sizes),nbr_p))

    total_guesses = np.zeros((len(sizes),nbr_p))
    
    P = np.zeros((len(sizes),nbr_p))


    x = [np.round(start_p + step_p*i, decimals=2) for i in range(nbr_p)] 

    #std = np.zeros((3,16))
    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(len(sizes)):
        size = sizes[i]
        for j in range(nbr_p):
            try:
                print()
                print('Processing...\nSize:', size)
                p = np.round(start_p + j * step_p, decimals=2)
                print('p_error:', p)

                file_suffix = 'data_size_' + str(size) + '_method_STRC_id_' + str(j) + '_perror_' + str(p) + '_steps_50625_2000err.xz'
                file_path = file_prefix + file_suffix
                print(file_path)

                iterator = MCMCDataReader(file_path, size)

                data = iterator.full()
                for k in range(int(len(data)/2)):
                    qubit_matrix = data[2*k].reshape(2,size,size)
                    eq_distr = data[2*k+1]

                    #print(qubit_matrix)
                    #print(eq_distr)

                    true_eq = _define_equivalence_class(qubit_matrix)
                    predicted_eq = np.argmax(eq_distr)

                    if predicted_eq == true_eq:
                        correct_guesses[i,j] += 1
                    total_guesses[i,j] += 1

                P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
                #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
                df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "nbr_failed": (total_guesses[i,j] - correct_guesses[i,j]), "method":'STRCL4'}), ignore_index=True)
            except:
                print('failed, moving on')
                continue
    
    print(df)
    df.to_pickle('data/fig7_STRC_15data_2000_L4.xz')

def create_PTEQ_2000err():
    file_prefix = 'data/new/'

    nbr_sizes = 1
    start_size = 5
    step_size = 2

    sizes = [5, 7, 9, 11, 13, 15, 17]

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02

    correct_guesses = np.zeros((len(sizes),nbr_p))

    total_guesses = np.zeros((len(sizes),nbr_p))
    
    P = np.zeros((len(sizes),nbr_p))


    x = [np.round(start_p + step_p*i, decimals=2) for i in range(nbr_p)] 

    #std = np.zeros((3,16))
    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(len(sizes)):
        size = sizes[i]
        for j in range(nbr_p):
            try:
                print()
                print('Processing...\nSize:', size)
                p = np.round(start_p + j * step_p, decimals=2)
                print('p_error:', p)

                file_suffix = 'data_size_' + str(size) + '_method_PTEQ_id_' + str(i*nbr_p + j) + '_perror_' + str(p) + '_fig7.xz'
                file_path = file_prefix + file_suffix
                print(file_path)

                iterator = MCMCDataReader(file_path, size)

                data = iterator.full()
                for k in range(int(len(data)/2)):
                    qubit_matrix = data[2*k].reshape(2,size,size)
                    eq_distr = data[2*k+1]

                    #print(qubit_matrix)
                    #print(eq_distr)

                    true_eq = _define_equivalence_class(qubit_matrix)
                    predicted_eq = np.argmax(eq_distr)

                    if predicted_eq == true_eq:
                        correct_guesses[i,j] += 1
                    total_guesses[i,j] += 1

                P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
                #std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
                #df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "nbr_failed": (total_guesses[i,j] - correct_guesses[i,j]), "method":'eMWPM'}), ignore_index=True)
                df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "method":'PTEQ'}), ignore_index=True)

            except:
                print('failed, moving on')
                continue
    
    print(df)
    df.to_pickle('data/fig7_15PTEQ_data_fig7.xz')

def create_MCMC_df_for_figure_7():
    file_prefix = 'data/'

    correct_guesses = np.zeros((3,16))
    total_guesses = np.zeros((3,16))
    P = np.zeros((3,16))
    x = [0.05 + 0.01*i for i in range(16)] 
    std = np.zeros((3,16))


    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(3):
        for j in range(16):
            size = (i + 1) * 2 + 1
            p = 0.05 + j * 0.01
            file_suffix = 'data_' + str(size) + 'x' + str(size) + '_p_' + str(p) + '.xz'

            file_path = file_prefix + file_suffix

            #iterator = MCMCDataReader('data/data_000.xz', 5)#file_path, size)
            iterator = MCMCDataReader(file_path, size)
            while iterator.has_next():
                qubit_matrix, eq_distr = iterator.next()
                true_eq = define_equivalence_class(qubit_matrix)
                predicted_eq = np.argmax(eq_distr)
                if predicted_eq == true_eq:
                    correct_guesses[i,j] += 1
                total_guesses[i,j] += 1
            print(j)
            P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
            std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
            df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "method":'MCMC'}))
    
    df.to_pickle('data/fig7_MCMC_data.xz')

def create_RL_df_for_figure_7():
    df = pd.DataFrame()
    x = [0.05 + 0.01*i for i in range(16)] 
    P_fusk = [0.9541, 0.9364, 0.9136, 0.8849, 0.8666, 0.8275, 0.7906, 0.7605, 0.7325, 0.6867, 0.6519, 0.6176, 0.5769, 0.5483, 0.5104, 0.4828]
    P = [0.95723333, 0.93936667, 0.91213333, 0.88676667, 0.85806667, 0.82906667, 0.79366667, 0.76386667, 0.72136667, 0.6871,     0.6507,     0.61116667, 0.57353333, 0.54103333, 0.51016667, 0.48093333]
    df = df.append(pd.DataFrame({"p":x, "d": '3x3', "P":P, "nbr_pts":10000, "method":'MCMC+DRL'}))
    #P = 
    #df = df.append(pd.DataFrame({"p":x, "d": '5x5', "P":P, "nbr_pts":10000, "method":'RL_hi'}))
    P = [0.9885, 0.9747, 0.9595, 0.9333, 0.9139, 0.8784, 0.8377, 0.7909, 0.7421, 0.6872, 0.6416, 0.5819, 0.5225, 0.4806, 0.4358, 0.382]
    df = df.append(pd.DataFrame({"p":x, "d": '5x5', "P":P, "nbr_pts":10000, "method":'MCMC+DRL'}))
    df.to_pickle('data/fig7_RL_data.xz')

def create_REF_df_for_figure_7():
    df = pd.DataFrame()
    x = [0.05 + 0.01*i for i in range(16)] 
    
    #P = [0.9556, 0.9411, 0.9169, 0.8878, 0.8639, 0.8192, 0.7868, 0.7547, 0.7243, 0.6789, 0.6392, 0.6053, 0.5703, 0.5286, 0.5016, 0.4625]
    #df = df.append(pd.DataFrame({"p":x, "d": '3x3', "P":P, "nbr_pts":10000, "method":'REF'}))

    P = [0.9909, 0.9814, 0.971, 0.9539, 0.9254, 0.8994, 0.8661, 0.827,  0.7772, 0.7247, 0.6882, 0.6258, 0.587,  0.5278, 0.4868, 0.4274]
    df = df.append(pd.DataFrame({"p":x, "d": '5x5', "P":P, "nbr_pts":10000, "method":'DRL'}))

    P = [0.9981, 0.9946, 0.9887, 0.9787, 0.9595, 0.9358, 0.9031, 0.8692, 0.8129, 0.7582, 0.6983, 0.6327, 0.5652, 0.4965, 0.4379, 0.3814]
    df = df.append(pd.DataFrame({"p":x, "d": '7x7', "P":P, "nbr_pts":10000, "method":'DRL'}))

    #P = [0.9483, 0.9216, 0.8986, 0.8654, 0.8413, 0.8054, 0.7601, 0.7294, 0.7013, 0.654, 0.6191, 0.5898, 0.5508, 0.5109, 0.4754, 0.4468]
    #df = df.append(pd.DataFrame({"p":x, "d": '9x9', "P":P, "nbr_pts":10000, "method":'REF'}))
    df.to_pickle('data/fig7_REF_data.xz')

'''def create_MWPM_df_for_figure_7():
    df = pd.DataFrame()
    #x = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.2] 
    x = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.2] 

    #P = [9.981500e-01,9.837100e-01,9.497900e-01,9.030200e-01,8.433500e-01,7.753300e-01,7.024700e-01,6.280200e-01,5.537600e-01,4.862800e-01,4.556400e-01]
    P = [9.497900e-01,9.030200e-01,8.433500e-01,7.753300e-01,7.024700e-01,6.280200e-01,5.537600e-01,4.862800e-01,4.556400e-01]
    df = df.append(pd.DataFrame({"p":x, "d": '3x3', "P":P, "nbr_pts":10000, "method":'MWPM'}))

    # lite större MWPM data.
    x = [0.05, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2] 
    P = [9.836900e-01,9.714800e-01,9.279500e-01,8.957500e-01,8.594800e-01,8.165100e-01,7.697700e-01,7.204300e-01,6.712200e-01,6.162400e-01,5.649100e-01,5.143300e-01,4.634000e-01,4.198500e-01,3.721000e-01]
    df = df.append(pd.DataFrame({"p":x, "d": '5x5', "P":P, "nbr_pts":10000, "method":'MWPM'}))
    
    x = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2] 
    P = [9.949200e-01,9.883500e-01,9.765200e-01,9.592800e-01,9.321700e-01,8.990400e-01,8.509200e-01,8.017900e-01,7.396600e-01,6.784800e-01,6.138100e-01,5.487400e-01,4.820700e-01,4.262300e-01,3.697100e-01,3.184300e-01]
    df = df.append(pd.DataFrame({"p":x, "d": '7x7', "P":P, "nbr_pts":10000, "method":'MWPM'}))

    df.to_pickle('data/fig7_eMWPM_data.xz')'''


def plot_fig7_1():
    # Time to plot stuff!
    df = pd.DataFrame()
    #df = df.append(pd.read_pickle('data/fig7_STDC_data_fig7.xz'))
    #df = df.append(pd.read_pickle('data/fig7_STDCrain_data_fig7.xz'))
    #df = df.append(pd.read_pickle('data/fig7_STDC_5*(5)*L4_data_hi_res.xz'))
    # df = df.append(pd.read_pickle('data/fig7_1884540.xz'))
    # df = df.append(pd.read_pickle('data/fig7_1884507.xz'))
    # df = df.append(pd.read_pickle('data/fig7_1884584.xz'))
    # df = df.append(pd.read_pickle('data/fig7_1884938.xz'))
    # df = df.append(pd.read_pickle('data/fig7_1890413.xz'))
    # df = df.append(pd.read_pickle('data/fig7_1890445.xz'))
    df = df.append(pd.read_pickle('data/fig7_1898458.xz'))
    df = df.append(pd.read_pickle('data/fig7_1898491.xz'))
    df = df.append(pd.read_pickle('data/fig7_1900780.xz'))
    df = df.append(pd.read_pickle('data/fig7_1900812.xz'))
    df = df.append(pd.read_pickle('data/fig7_1900844.xz'))
    df = df.append(pd.read_pickle('data/fig7_1900876.xz'))

    #df = df.append(pd.read_pickle('data/fig7_eMWPM_15data_fig7.xz'))
    #df = df.append(pd.read_pickle('data/fig7_15PTEQ_data_fig7.xz'))
    
    df.columns = ['p', 'Size, $L$', 'P', 'nbr_pts', 'Method']

    #df = df.loc[df.p > 0.10]


    # get error plot
    #df['P'] = df['P'].apply(lambda x: 1 - x)

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    ax = sns.lineplot(x='p', y='P', hue='Size, $L$', style='Method', data=df)
    #ax.set(xscale="log", yscale="log")
    #ax.set(xlim=(None, 0.1))
    #ax.set(ylim=(0.8, 1))
    
    ax.set_xlabel("Physical error rate, $p$",fontsize=15)
    ax.set_ylabel("Logical correction rate, $P_s$",fontsize=15)
    ax.tick_params(labelsize=12)

    #ax.set_yscale('log')
    #ax.set_xscale('log')

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    #plt.tight_layout()
    plt.savefig('plots/benchmark_PTEQvsSTDC_equalstepsV2.pdf')

def plot_fig7_1_psamp():
    # Time to plot stuff!
    #df = pd.read_pickle('data/fig7_ALL_data.xz')
    df = pd.read_pickle('data/fig7_ALL_data_11x11_psamp.xz')
    #df = df.append(pd.read_pickle('data/fig7_PTEQ_data.xz'))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    

    #df = df.append(pd.read_pickle('data/fig7_RL_data.xz'))
    #df = df.append(pd.read_pickle('data/fig7_REF_data.xz'))
    #df = df.append(pd.read_pickle('data/fig7_MWPM_data.xz'))

    df.columns = ['p', '$p_{sample}$', 'P', 'nbr_pts', 'Method']

    #df = df.loc[df.Method == 'STDC']
    #df = df.loc[df.p < 0.1]
    #df = df.loc[df.p > 0.14]

    # get error plot
    #df['P'] = df['P'].apply(lambda x: 1 - x)

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    ax = sns.lineplot(x='p', y='P', hue='$p_{sample}$', style='Method', palette=sns.color_palette("cubehelix", 6), data=df, legend='full')
    #ax.set(xscale="log", yscale="log")
    #ax.set(xlim=(None, 0.1))
    #ax.set(ylim=(0.8, 1))
    
    #ax.set_title("11x11")
    ax.set_xlabel("Physical error rate, $p$",fontsize=15)
    ax.set_ylabel("Logical correction rate, $P_s$",fontsize=15)
    ax.tick_params(labelsize=12)

    #ax.set_yscale('log')
    #ax.set_xscale('log')

    plt.savefig('plots/testingpsamp.pdf')

def plot_fig7_1_ndrop():
    # Time to plot stuff!
    #df = pd.read_pickle('data/fig7_ALL_data.xz')
    df = pd.read_pickle('data/fig7_ALL_data_15x15_ndrop.xz')
    #df = df.append(pd.read_pickle('data/fig7_PTEQ_data.xz'))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    

    #df = df.append(pd.read_pickle('data/fig7_RL_data.xz'))
    #df = df.append(pd.read_pickle('data/fig7_REF_data.xz'))
    #df = df.append(pd.read_pickle('data/fig7_MWPM_data.xz'))

    df.columns = ['p', '$N_{drop}$', 'P', 'nbr_pts', 'Method']

    #df = df.loc[df.Method == 'STDC']
    #df = df.loc[df.p < 0.1]
    #df = df.loc[df.p > 0.14]

    # get error plot
    #df['P'] = df['P'].apply(lambda x: 1 - x)

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    ax = sns.lineplot(x='p', y='P', hue='$N_{drop}$', style='Method', palette=sns.color_palette("cubehelix", 4), data=df, legend='full')
    #ax.set(xscale="log", yscale="log")
    #ax.set(xlim=(None, 0.1))
    #ax.set(ylim=(0.8, 1))
    
    #ax.set_title("11x11")
    ax.set_xlabel("Physical error rate, $p$",fontsize=15)
    ax.set_ylabel("Logical correction rate, $P_s$",fontsize=15)
    ax.tick_params(labelsize=12)

    #ax.set_yscale('log')
    #ax.set_xscale('log')

    plt.savefig('plots/testingndrop15.pdf')

def plot_relfail():
    # Time to plot stuff!

    # Read baseline and PTEQ
    # L = 15 data
    '''df = pd.read_pickle('data/fig7_eMWPM_15data_2000.xz')
    df = df.append(pd.read_pickle('data/fig7_MWPM_15data_2000.xz'), ignore_index=True)
    df = df.append(pd.read_pickle('data/fig7_STDC_15data_2000_L4.xz'), ignore_index=True)
    df = df.append(pd.read_pickle('data/fig7_ST_15data_2000_L4.xz'), ignore_index=True)'''
    #df = df.append(pd.read_pickle('data/fig7_STRC_15data_2000_L4.xz'), ignore_index=True)
    #df = df.append(pd.read_pickle('data/fig7_15PTEQ_data_2000.xz'), ignore_index=True)
    #df = df.append(pd.read_pickle('data/fig7_PTDC_15data_2000.xz'), ignore_index=True)

    # L=9 data
    df = pd.read_pickle('data/fig7_eMWPM_data_2000.xz')
    df = df.append(pd.read_pickle('data/fig7_MWPM_data_2000.xz'), ignore_index=True)
    df = df.append(pd.read_pickle('data/fig7_STDC_data_2000_L4.xz'), ignore_index=True)
    df = df.append(pd.read_pickle('data/fig7_ST_data_2000_L4.xz'), ignore_index=True)
    

    df.columns = ['p', 'Size', 'P', 'nbr_pts', 'nbr_failed', 'Method']
    df = df.loc[df.Size == '9x9']

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    df_stdc = df.loc[df.Method == 'STDC']
    df_strc = df.loc[df.Method == 'STRC']
    df_pteq = df.loc[df.Method == 'PTEQ']
    df_emwpm = df.loc[df.Method == 'eMWPM']
    df_mwpm = df.loc[df.Method == 'MWPM']

    '''
    df_stdc['test'] = (1-df_mwpm['P']) / (1 - df_stdc['P'])
    df_strc['test'] = (1-df_mwpm['P']) / (1-df_strc['P'])
    df_pteq['test'] = (1-df_mwpm['P']) / (1-df_pteq['P'])
    df_mwpm['test'] = (1-df_mwpm['P']) / (1-df_mwpm['P'])
    df_emwpm['test'] = (1-df_mwpm['P']) / (1-df_emwpm['P'])
    '''

    df_mwpm['relF'] = (1-df_mwpm['P']) / (1-df_mwpm['P'])
    df_out = pd.DataFrame()
    df_out = df_out.append(df_mwpm, ignore_index=True)
    
    #print('here')
    #print(df_out)

    nbr_p = 8
    start_p = 0.05
    step_p = 0.02


    for j in range(nbr_p):
        p = np.round(start_p + step_p * j, decimals=2)

        try:
            tempdf = df.loc[df.p == p].loc[df.Method == 'eMWPM']
            if tempdf.nbr_failed.values > 1:
                tempdf['relF'] = (1-df.loc[df.p == p].loc[df.Method == 'eMWPM']['P'].values) / (1-df.loc[df.p == p].loc[df.Method == 'MWPM']['P'].values)
                df_out = df_out.append(tempdf, ignore_index=True)
        except:
            print('failed!')
        
        try:
            tempdf = df.loc[df.p == p].loc[df.Method == 'STDCL4']
            if tempdf.nbr_failed.values > 1:
                tempdf['relF'] = (1-df.loc[df.p == p].loc[df.Method == 'STDCL4']['P'].values) / (1-df.loc[df.p == p].loc[df.Method == 'MWPM']['P'].values)
                df_out = df_out.append(tempdf, ignore_index=True)
        except:
            print('failed!')

        try:
            tempdf = df.loc[df.p == p].loc[df.Method == 'STL4']
            if tempdf.nbr_failed.values > 1:
                tempdf['relF'] = (1-df.loc[df.p == p].loc[df.Method == 'STL4']['P'].values) / (1-df.loc[df.p == p].loc[df.Method == 'MWPM']['P'].values)
                df_out = df_out.append(tempdf, ignore_index=True)
        except:
            print('failed!')

        try:
            tempdf = df.loc[df.p == p].loc[df.Method == 'STRCL4']
            if tempdf.nbr_failed.values > 1:
                tempdf['relF'] = (1-df.loc[df.p == p].loc[df.Method == 'STRCL4']['P'].values) / (1-df.loc[df.p == p].loc[df.Method == 'MWPM']['P'].values)
                df_out = df_out.append(tempdf, ignore_index=True)
        except:
            print('failed!')

        try:
            tempdf = df.loc[df.p == p].loc[df.Method == 'PTEQ']
            if tempdf.nbr_failed.values > 1:
                tempdf['relF'] = (1-df.loc[df.p == p].loc[df.Method == 'PTEQ']['P'].values) / (1-df.loc[df.p == p].loc[df.Method == 'MWPM']['P'].values)
                df_out = df_out.append(tempdf, ignore_index=True)
        except:
            print('failed!')

        try:
            tempdf = df.loc[df.p == p].loc[df.Method == 'PTDC']
            if tempdf.nbr_failed.values > 1:
                tempdf['relF'] = (1-df.loc[df.p == p].loc[df.Method == 'PTDC']['P'].values) / (1-df.loc[df.p == p].loc[df.Method == 'MWPM']['P'].values)
                df_out = df_out.append(tempdf, ignore_index=True)
        except:
            print('failed!')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df_out)

    
    #df_out['relF'] = df_out['relF'].apply(lambda x: np.log(x))
    

    #cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    ax = sns.lineplot(x='p', y='relF', hue='Method', palette=sns.color_palette("cubehelix", 4),marker="o", data=df_out)
    #ax = sns.lineplot(x='p', y='relF', hue='Method', data=df_out)
    #ax.set(yscale="log")
    ax.set_yscale('log')
    #ax.set(xlim=(None, 0.1))
    #ax.set(ylim=(0.8, 1))


    ax.set_xlabel("Physical error rate, $p$",fontsize=15)
    ax.set_ylabel("Relative logical error rate",fontsize=15)
    ax.tick_params(labelsize=12)

    plt.savefig('plots/benchmark_L_9_2000err_L4_log.pdf')

def binom(n, k):
    return factorial(n) // factorial(k) // factorial(n - k)


def plot_threshold_lines():
    # Time to plot stuff!
    df = pd.DataFrame()

    #df = pd.read_pickle('data/fig7_STDC_data_fig7_alexei.xz')

    #df = df.append(pd.read_pickle('data/fig7_STDC_5*(5)*L4_data_hi_res.xz'))
    df = df.append(pd.read_pickle('data/fig7_STDC_5*(5)*L5_data_hi_res.xz'))
    df = df.append(pd.read_pickle('data/fig7_STDC_5*(5)*L5_data_hi_res_extra.xz'))
    df = df.append(pd.read_pickle('data/fig7_STDC_5*(5)*L5_data_hi_res_extra2.xz'))
    
    #df = df.append(pd.read_pickle('data/fig7_15PTEQ_data_fig7.xz'))

    df.columns = ['p', 'Size, $L$', 'P', 'nbr_pts', 'Method']

    df.replace({"5x5":5,"7x7":7,"9x9":9,"11x11":11,"13x13":13,"15x15":15,"17x17":17,"19x19":19,"21x21":21,"23x23":23,"25x25":25}, inplace=True)
    
    df = df.loc[df['p'] < 0.192]
    df = df.loc[df['p'] > 0.169]
    #df = df.loc[df['Size, $L$'] > 8]
    
    # Give me my indicies back!
    df = df.reset_index(drop=True)

    # concaternate identical datapoints
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    seen_data = set()

    for index, row in df.iterrows():
        datapoint_key = str(row['p']) + str('-') + str(row['Size, $L$'])
        if datapoint_key not in seen_data:
            seen_data.add(datapoint_key)
        else: # This is where concaternation happens
            size = row['Size, $L$']
            p = row['p']
            new_P = row['P']
            new_nbr_pts = row['nbr_pts']


            old_df = df[(df['p'] == p) & (df['Size, $L$'] == size)].iloc[0]
            old_P = old_df['P']
            old_nbr_pts = old_df['nbr_pts']
            old_index = old_df.name

            updated_nbr_pts = old_nbr_pts + new_nbr_pts
            updated_P = (old_P*old_nbr_pts + new_P*new_nbr_pts) / updated_nbr_pts
            print('index', old_index, 'other', updated_nbr_pts, updated_P)
            df.at[old_index,'nbr_pts'] = updated_nbr_pts
            df.at[old_index,'P'] =  updated_P

            df.drop(index=index, inplace=True)

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #    print(df)

    #df['P'] = df['P'].apply(lambda x: np.log(1 - x))



    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


    #cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    #ax = sns.lineplot(x='Size, $L$', y='P', hue='p', style='Method', data=df, legend='full', palette=sns.color_palette("Paired", 32))
    fig = plt.figure()
    ax = fig.add_subplot()
    print(np.array(df["p"]))

    for i in range(5):
        ax.errorbar(np.array(df["Size, $L$"]).reshape((11,-1))[:,i], np.array(df["P"]).reshape((11,-1))[:,i], np.sqrt(np.array(df["P"]).reshape((11,-1))[:,i]*(1-np.array(df["P"]).reshape((11,-1))[:,i])/np.array(df["nbr_pts"]).reshape((11,-1))[:,i]), linestyle='--', label=f"p = {np.array(df['p'])[i]}")
        #ax.plot(np.array(df["Size, $L$"]).reshape((11,-1))[:,i], np.array(df["P"]).reshape((11,-1))[:,i], '.--', label=f"p = {np.array(df['p'])[i]}")

    #print(plt.gca().get_lines())
    labelLines(plt.gca().get_lines(), zorder=2.5, xvals=np.full(6, 11))
    
    ax.set_xlabel("Size, $L$",fontsize=15)
    #ax.set_ylabel("Logical correction rate, $P_s$",fontsize=15)
    ax.set_ylabel(r"Logarithmized failure rate, $\log{P_f}$",fontsize=15)
    ax.tick_params(labelsize=12)
    #plt.legend(loc='right', fontsize='xx-small')
    #plt.ylim((-2,-1))

    #ax.set_yscale('log')
    #ax.set_xscale('log')

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    plt.savefig('plots/benchmark_STDC_5(5)L5_hi_res_PsL_log.pdf')
    #plt.savefig('plots/benchmark_test.pdf')


if __name__ == '__main__':

    '''normal plotting stuff'''
    #create_eMWPM_df_for_figure_7()
    #create_PTEQ_2000err()
    #create_df_for_figure_7()
    #create_fromnewstructure_comp()

    #create_ALL_df_for_figure_7()
    #plot_fig7_1()

    '''rel plotting stuff'''
    #create_PTDC_df_for_figure_7()
    #create_eMWPM_df_for_figure_7()
    #create_MWPM_df_for_figure_7()
    #create_STDC_2000err()
    #create_ST_2000err()
    #create_STRC_2000err()
    #create_PTEQ_2000err()
    #plot_relfail()

    '''psample plotting stuff'''
    #create_ALL_df_for_figure_7_psample()
    #plot_fig7_1_psamp()


    '''ndrop plotting stuff'''
    #create_ALL_df_for_figure_7_ndrop()
    #plot_fig7_1_ndrop()

    '''Plot threshold lines'''
    #create_extra()

    #create_fromnewstructure()

    #plot_threshold_lines()
    #plot_p()
    #N_n()
    # N_n_expbetan()
    #N_n_3d()
    plot_Nobs()
    pass