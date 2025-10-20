import numpy as np
import torch
from argparse import ArgumentParser
from helpers import _create_filter, _reshape_tensor


def explicit_svd(t, n, compute_uv=False):
    """
    Compute the singular values (and optionally singular vectors) of a convolutional layer by constructing its explicit matrix representation.

    Parameters
    ----------
    t: torch.nn.Conv2d
        The convolutional layer whose SVD is computed.
    n: int
        Spatial dimension (height and width) of the input image/tensor.
    compute_uv: bool, optional (default=False)
        Whether to compute singular vectors (U, Vh) in addition to the singular values.
    Returns
    -------
    svs: tuple or np.ndarray
        - If compute_uv is False: returns a 1D NumPy array of singular values.
        - If compute_uv is True: returns (U, S, Vh) from `np.linalg.svd`.

    Notes
    -----
    - The explicit matrix representation is built via `reshape_tensor()`.
    - This method is computationally expensive and should only be used for small spatial dimensions (e.g., n ≤ 16).
    """

    c = t.weight.shape[1]
    A_matrix = _reshape_tensor(t, c, n)
    svs = np.linalg.svd(A_matrix, compute_uv=compute_uv)

    return svs

def svd_lfa(A, n, compute_uv=False):
    """
    Compute the singular values (and optionally singular vectors) of a convolutional layer by local Fourier analysis.
    Parameters
    ----------
    A: np.ndarray or torch.Tensor
        The convolutional layer whose SVD is computed.
    n: int
        Spatial dimension (height and width) of the input image/tensor.
    compute_uv: bool, optional (default=False)
        Whether to compute singular vectors (U, Vh) in addition to the singular values.
    Returns
    -------
    svs: tuple or np.ndarray
        - If compute_uv is False: returns a 1D NumPy array of singular values.
        - If compute_uv is True: returns (U, S, Vh) from `np.linalg.svd`.
    """

    # Convert to NumPy
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()

    # Frequency grid normalized to [0,1)
    k = np.linspace(0, 1, n + 1)[:-1]
    K = np.array([-1 * k, 0 * k, 1 * k])

    # 1D complex exponentials
    p = np.exp(2 * 1j * np.pi * K)

    # Compute the Fourier transform of the kernel by applying appropriate phase shifts along both spatial axes
    d = np.tensordot(A.astype(np.complex128), p, axes=1)
    H = np.tensordot(d, p, axes=([-2, 0]))

    # Move frequency axes first so np.linalg.svd treats the last two dims as matrices
    # Final shape: (n, n, c_out, c_in)
    r = H.transpose((2, 3, 0, 1))

    # Batched SVD over the last two dims
    svs = np.linalg.svd(r, compute_uv=compute_uv)

    return svs


def main(_):
    c = args.c
    n = args.n
    method = args.m
    uv=args.uv

    print('unrolled matrix dimension:', c * (n ** 2), '*', c * (n ** 2))
    A, T = _create_filter(c)

    if method in ['e', 'eonly']:
        if n >= 256:
            print("⚠ Explicit SVD is too large for n ≥ 256 — use method='lfa' instead.")
            return
        svs = explicit_svd(T, n, compute_uv=uv)
        print(f"Explicit calculation complete. Spectral radius: {np.max(np.abs(svs)):.6f}")
        if method == 'eonly':
            return
        else:
            lfa_svs = svd_lfa(A, n, compute_uv=uv)
            print(f"LFA calculation complete. Spectral radius: {np.max(np.abs(lfa_svs)):.6f}")

    elif method == 'lfa':
        lfa_svs = svd_lfa(A, n, compute_uv=uv)
        print(f"Calculation complete via LFA. Spectral radius: {np.max(np.abs(lfa_svs)):.6f}")

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'e' (explicit) or 'lfa'.")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--m", "--method", default='lfa', choices=['lfa', 'e', 'eonly'], type=str)
    parser.add_argument("--c", "--number of channels", default=16, type=int)
    parser.add_argument("--n", "--input size", default=128, type=int)
    parser.add_argument("--uv", "--compute singular vectors", default=False, type=bool)

    args = parser.parse_args()

    main(args)
