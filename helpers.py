import numpy as np
import torch
import torch.nn as nn

def _create_filter(c):
    """
    Create a 2D convolutional layer with periodic (circular) padding.

    Parameters
    ----------
    c: int
        Number of input and output channels (the layer is square: c_in=c_out=c)
    Returns
    -------
    weight_array: np.ndarray
        The convolutional weights as a NumPy array of shape (c_in, c_out, c_out, c).
    layer: nn.Module
        The corresponding layer  with circular padding.

    Notes
    -----
    - This setup enforces periodic boundary conditions, consistent with local Fourier analysis.
    """

    c_in = c
    c_out = c_in
    s = 3
    padding = 1
    padding_mode = 'circular'
    layer = nn.Conv2d(c_in, c_out, kernel_size=s, stride=1,
                  padding=padding, padding_mode=padding_mode,
                  bias=False, groups=1)
    weight_array = layer.weight.detach().cpu().numpy()
    return weight_array, layer

def _create_tensors_bc(c_in, c_out):
    """
    Create 2D convolutional layers with identical weights with periodic (circular) padding and zero padding.

    Parameters
    ----------
    c_in: int
        Number of input channels.
    c_out: int
        Number of output channels.

    Returns
    -------
    Z: torch.nn.Conv2d
        Convoluted 2D convolutional layer with zero padding.
    P: torch.nn.Conv2d
        torch.nn.Conv2d
    """

    c_in = c_in
    c_out = c_out
    s = (3, 3)
    padding = 1

    Z = nn.Conv2d(c_in, c_out, kernel_size=s, stride=1,
                  padding=padding, padding_mode='zeros',
                  bias=False, groups=1)
    P = nn.Conv2d(c_in, c_out, kernel_size=s, stride=1,
                  padding=padding, padding_mode='circular',
                  bias=False, groups=1)

    P.weight = Z.weight

    return Z, P, c_in, c_out


def _reshape_tensor(t, c_in, n):
    """
     Construct the explicit matrix representation of a 2D convolutional layer.

     Parameters
     ----------
     t: torch.nn.Conv2d
        The convolutional layer whose action is to be represented as a matrix.
        The resulting matrix reflects the layer's own padding and stride settings
     c_in: int
        Number of input channels.
     n: int
        Spatial dimension (height and width) of the input image/tensor.

     Returns
     -------
     A : torch.Tensor
        The explicit linear operator matrix of shape (c_out * n^2, c_in * n^2),
        where applying `A @ x.flatten()` is equivalent to applying the convolution.
     Notes
     -----
     This method applies the convolution to each canonical basis vector to recover the full operator structure.
     """

    c_out = c_in
    rank = min(c_in, c_out)

    os = torch.eye(rank * n * n)
    l = []
    for i in os:
        l.append(t(i.view([1, rank, n, n])).view(-1).detach().cpu())
    A = torch.stack(l)
    A = A.T
    return A

