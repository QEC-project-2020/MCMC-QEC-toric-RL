import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import from_numpy
from collections import namedtuple
import pandas as pd


Action = namedtuple('Action', ['position', 'action'])

Perspective = namedtuple('Perspective', ['perspective', 'position'])

Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'terminal'])


def conv_to_fully_connected(input_size, filter_size, padding, stride):
    return (input_size - filter_size + 2 * padding)/ stride + 1


def pad_circular(x, pad):
    x = torch.cat([x, x[:,:,:,0:pad]], dim=3)
    x = torch.cat([x, x[:,:, 0:pad,:]], dim=2)
    x = torch.cat([x[:,:,:,-2 * pad:-pad], x], dim=3)
    x = torch.cat([x[:,:, -2 * pad:-pad,:], x], dim=2)
    return x


def incremental_mean(x, mu, N):
    return mu + (x - mu) / (N)


def convert_from_np_to_tensor(tensor):
    tensor = from_numpy(tensor)
    tensor = tensor.type('torch.Tensor')
    return tensor


# Returns number of degenerate equivalence classes in eq_distr given a relative error tolerance rel_tol
def count_degenerate(eq_distr, rel_tol=0.1):
    sorted_distr = np.sort(eq_distr)[::-1]

    i = 0
    while sorted_distr[i+1] > sorted_distr[i] * (1 - rel_tol):
        i += 1
    
    return i + 1


# Helper function. 
# Returns an int representing logical operators needed to reach 
# equivalence classes eqs from class 0
def eq_to_ops(eqs):
    # eq_class is a 4-digit binary number (z2 x2 z1 x1)
    # return is two 2-digit binary numbers (op2 op1), where op=0,1,2,3 corresponding to logical operators in layers 2 and 1
    # flip x2 if z2==1, flip x1 if z1==1
    mask = 0b1010
    xor = (mask & eqs) >> 1
    ops = eqs ^ xor
    ops2 = ops >> 2
    ops1 = 0b0011 & ops
    return np.array([ops1, ops2]).transpose()


# Returns a 16-by-2 array of logical operators to apply to eq 
# to get a list of error chains ordered by equivalence class
def eq_to_ordered_ops(eq):
    # Calculate for eq==0
    ops = eq_to_ops(np.arange(16))
    # Add/remove operators between eq and (eq==0)
    ops = ops ^ ops[eq]

    # Split the 4-digit binary into two logical operators (layers 1, 2)
    #ops2 = ops >> 2
    #ops1 = 0b0011 & ops
    #return np.stack((ops1, ops2), axis=1)
    return ops
