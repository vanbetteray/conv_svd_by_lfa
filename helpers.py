# helper functions
import numpy as np
import torch
import torch.nn as nn

def _create_filter(c):
    """
    creates a pytorch 2D convolutional layer
    :param c: number of channels
    :return: weights as array, as torch tensor
    """
    c_in = c
    c_out = c_in
    s = 3
    padding = 1
    padding_mode = 'circular'
    T = nn.Conv2d(c_in, c_out, kernel_size=s, stride=1,
                  padding=padding, padding_mode=padding_mode,
                  bias=False, groups=1)
    return np.array(T.weight.detach()), T


def _reshape_tensor(t, c_in, m):
    """
    :param t: weight tensor
    :param c_in: number if input channls
    :param m: number if colums/rows
    :return:
    """
    c_out = c_in
    rank = min(c_in, c_out)

    os = torch.eye(rank * m * m)
    l = []
    for i in os:
        l.append(t(i.view([1, rank, m, m])).view(-1).detach().cpu())
    A = torch.stack(l)
    A = A.T
    return A
