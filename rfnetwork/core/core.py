import numpy as np
from pathlib import Path
from np_struct import ldarray
import os

import numpy as np

from scipy.interpolate import CubicSpline
from . units import const, conv
from . import core_func


def junction_sdata(frequency: np.ndarray, N: int) -> np.ndarray:
    sdata = np.full((N, N), 2 / N, dtype="complex128")
    np.fill_diagonal(sdata, (2 / N) - 1)
    return np.broadcast_to(sdata, (len(frequency), N, N)).copy()
    

def connect(c1: dict, c2: dict, connections: list, probes: list = None, n_threads: int = None):
    """
    Connect multiple ports between two components s1 and s2.
    Connections is of the form [(s1 port, s2 port), (s1 port, s2 port), ...].
    """

    # automatically set the number of threads to half the cpu count. 
    if n_threads is None:
        n_cpu = os.cpu_count()
        n_threads = n_cpu // 2 if n_cpu is not None and n_cpu > 2 else 1

    connections = np.array(np.atleast_2d(connections), dtype=np.int32, order="C")

    if probes is not None:
        probes = np.array(np.atleast_2d(probes), dtype=np.int32, order="C")
    else:
        probes = np.zeros_like(connections, dtype=np.int32, order="C")

    n_connections = len(connections)

    noise = "n" in c1.keys()
    
    f_len, s1_b, s1_a = c1["s"].shape
    f_len, s2_b, s2_a = c2["s"].shape

    a_len = s1_a + s2_a - (2 * n_connections)
    b_len = s1_b + s2_b
    n_row = b_len - (2 * n_connections) + np.count_nonzero(probes)

    row_order = np.zeros(n_row, dtype=np.int32, order="C")

    cas_s = np.zeros((f_len, n_row, a_len), dtype="complex128")
    cas_n, n1, n2 = None, None, None

    s1, s2 = np.ascontiguousarray(c1["s"]), np.ascontiguousarray(c2["s"]) 

    if noise:
        cas_n = np.zeros((f_len, a_len, a_len), dtype="complex128")
        n1 = np.ascontiguousarray(c1["n"])
        n2 = np.ascontiguousarray(c2["n"])

    core_func.connect_other(s1, s2, n1, n2, connections, probes, row_order, cas_s, cas_n, int(n_threads))

    cas = dict(s=cas_s)

    if noise:
        cas["n"] = cas_n

    return row_order, cas


def connect_self(c1: dict, connections: list, probes: list = None, n_threads: int = None):
    """
    Connect multiple ports between two components s1 and s2.
    Connections is of the form [(s1 port, s2 port), (s1 port, s2 port), ...].
    """

    # automatically set the number of threads to half the cpu count. 
    if n_threads is None:
        n_cpu = os.cpu_count()
        n_threads = n_cpu // 2 if n_cpu is not None and n_cpu > 2 else 1

    connections = np.array(np.atleast_2d(connections), dtype=np.int32, order="C")

    if probes is not None:
        probes = np.array(np.atleast_2d(probes), dtype=np.int32, order="C")
    else:
        probes = np.zeros_like(connections, dtype=np.int32, order="C")

    n_connections = len(connections)

    noise = "n" in c1.keys()
    
    f_len, s1_b, s1_a = c1["s"].shape

    a_len = s1_a - (2 * n_connections)
    b_len = s1_b

    n_row = b_len - (2 * n_connections) + np.count_nonzero(probes)
    row_order = np.zeros(n_row, dtype=np.int32, order="C")

    cas_s = np.zeros((f_len, n_row, a_len), dtype="complex128")
    cas_n, n1 = None, None

    if noise:
        cas_n = np.zeros((f_len, a_len, a_len), dtype="complex128")
        n1 = np.ascontiguousarray(c1["n"])

    s1 = np.ascontiguousarray(c1["s"])

    core_func.connect_self(s1, n1, connections, probes, row_order, cas_s, cas_n, int(n_threads))

    cas = dict(s=cas_s)

    if noise:
        cas["n"] = cas_n

    return row_order, cas


def noise_params_to_ndata(np_data: np.ndarray, sdata: np.ndarray):
    """
    Converts noise parameters into a 2x2 noise correlation matrix.
    Frequency vectors of np_data and sdata must be identical.
    """

    N = sdata.shape[-1]
    NFREQ = sdata.shape[-3]

    if N != 2:
        raise NotImplementedError("Noise parameters are only supported for 2-port networks.")

    nf_min, gmma_opt, rn = np_data.T

    # initialize noise wave matrix
    ndata = np.zeros(shape=(NFREQ, 2, 2), dtype="complex128")

    # constants, rn is already normalized.
    t = (4 * rn * const.t0)

    # pull out sdata for better readability
    s11 = sdata[:, 0,0]
    s21 = sdata[:, 1,0]

    # convert nfmin to temperature
    tmin = const.t0 * (nf_min - 1)

    c2 = const.k*(np.abs(s21)**2) * (tmin + t * ((np.abs(gmma_opt)**2)/(np.abs(1 + gmma_opt)**2)))
    c1 = const.k*tmin * (np.abs(s11)**2  -1) + ((const.k*t*np.abs(1-(s11*gmma_opt))**2)/(np.abs(1+gmma_opt)**2))
    c1c2 = ((-np.conj(s21)*np.conj(gmma_opt)*const.k*t) / ( np.abs(1 + gmma_opt)**2)) + ((s11/s21)*c1)
    c2c1 = np.conj(c1c2)

    ndata[:, 0,0] = c1
    ndata[:, 1,1] = c2
    ndata[:, 0,1] = c1c2
    ndata[:, 1,0] = c2c1

    return ndata

def get_passive_ndata(sdata: np.ndarray):
    """
    Correlation matrix for a passive network. Passivity is not verified.
    """
    # https://authors.library.caltech.edu/995/1/WEDieeemgwl91.pdf
    
    N = sdata.shape[-1]
    idn = np.identity(N, dtype="complex128")
    idn = np.broadcast_to(idn, sdata.shape).copy()

    s_conj_T = np.transpose(sdata, (-3, -1, -2)).conj()
    return (idn - np.einsum("...ij, ...jk->...ik", sdata, s_conj_T)) * const.k * const.t0


def is_passive(sdata: np.ndarray):
    return np.max(np.abs(sdata)) < 1.01


def noise_figure_from_ndata(sdata: np.ndarray, ndata: np.ndarray, path: tuple, t0: float = 290):
    """
    Linear noise figure.
    """
    # path gain
    s21 = sdata[:, path[0] - 1, path[1] - 1]
    s21_m = s21 * s21.conj()
    # output noise power of output port
    nout = np.abs(ndata[:, path[0] - 1, path[0] - 1])
    # device equivalent temperature
    te = nout / (const.k * s21_m)
    # noise figure
    return (te / t0) + 1