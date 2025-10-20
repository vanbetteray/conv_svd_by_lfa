import numpy as np
from argparse import ArgumentParser
from helpers import _create_filter, _reshape_tensor


def explicit_svd(t, n, uv):
    """
    :param t: weight matrix
    :param n: number of singular values
    :param uv: bool: calculate singular values
    :return:
    """
    c = t.weight.shape[1]
    A_matrix = _reshape_tensor(t, c, n)
    svs = np.linalg.svd(A_matrix, compute_uv=uv)

    return svs

def svd_lfa(A, n, uv):
    """
    :param A: weight matrix
    :param n: int: number of singular values
    :param uv: bool: calculate singular vectors
    :return: singular values

    """
    k = np.linspace(0, 1, n + 1)[:-1]
    K = np.array([-1 * k, 0 * k, 1 * k])
    p = np.exp(2 * 1j * np.pi * K)
    d = np.tensordot(A, p, axes=1)
    d2 = np.tensordot(d, p, axes=([-2, 0]))
    r = d2.transpose((2, 3, 0, 1))
    svs = np.linalg.svd(r, compute_uv=uv)
    return svs


def main(_):
    c = args.c
    n = args.n
    method = args.method

    print('unrolled matrix dimension:', c * (n ** 2), '*', c * (n ** 2))
    A, T = _create_filter(c)

    if n < 256 and method == 'e':
        svs = explicit_svd(T, n, uv=False)
        print('max sv', max(svs))

    elif method == 'lfa':
        lfa_svs = svd_lfa(A, n, uv=False)
        print('max sv', max(lfa_svs.flatten()))

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--method", "--method", default='lfa', type=str)
    parser.add_argument("--c", "--number of channels", default=16, type=int)
    parser.add_argument("--n", "--input size", default=128, type=int)
    args = parser.parse_args()

    main(args)
