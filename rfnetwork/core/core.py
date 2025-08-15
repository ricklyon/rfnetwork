import numpy as np
from pathlib import Path
from np_struct import ldarray

import numpy as np

from scipy.interpolate import CubicSpline
from . units import const, conv
from . import core_func


def junction_sdata(frequency: np.ndarray, N: int) -> np.ndarray:
    sdata = np.full((N, N), 2 / N, dtype="complex128")
    np.fill_diagonal(sdata, (2 / N) - 1)
    return np.broadcast_to(sdata, (len(frequency), N, N)).copy()
    

def connection_matrix(s1: np.ndarray, s2: np.ndarray, connections: list, probes: list = None):

    connections = np.atleast_2d(connections)

    if probes is None:
        probes = np.zeros(connections.shape)
    else:
        probes = np.atleast_2d(probes)

    connections = np.array(connections, dtype=np.int32, order="C")
    probes = np.array(probes, dtype=np.int32, order="C")

    s1_b, s1_a = s1.shape[-2:]
    s2_b, s2_a = s2.shape[-2:]

    b_len = s1_b + s2_b
    a_len = s1_a + s2_a
    f_len = s1.shape[-3] # number of frequencies

    m1_shape = (f_len, b_len, b_len)
    m2_shape = (f_len, b_len, a_len - (len(connections) * 2))

    # create first matrix, starts with just the identity matrix
    m1 = np.zeros(m1_shape, dtype="complex128")
    m2 = np.zeros(m2_shape, dtype="complex128")

    row_order = np.zeros(b_len, dtype=np.int32, order="C")

    core_func.connection_matrix(s1, s2, connections, probes, m1, m2, row_order)

    n_rows = np.argmax(row_order == -1) if -1 in row_order else len(row_order)
    return m1[:, :n_rows], m2, row_order[:n_rows]

def connection_matrix_py(s1: np.ndarray, s2: np.ndarray, connections: np.ndarray, probes: np.ndarray = None):
    """
    Connect multiple ports between two components s1 and s2.
    Connections is of the form [(s1 port, s2 port), (s1 port, s2 port), ...].
    """
    connections = np.atleast_2d(connections)
    
    if probes is None:
        probes = np.zeros(connections.shape)
    else:
        probes = np.atleast_2d(probes)

    s1_b, s1_a = s1.shape[-2:]
    s2_b, s2_a = s2.shape[-2:]

    b_len = s1_b + s2_b
    a_len = s1_a + s2_a
    m1_shape = (s1.shape[-3], b_len, b_len)
    m2_shape = (s2.shape[-3], b_len, a_len)

    # create first matrix, starts with just the identity matrix
    m1 = np.identity(b_len, dtype="complex128")
    m1 = np.broadcast_to(m1, m1_shape).copy()

    # create second matrix, this is a "block" diagonal matrix with S1, S2 down the diagonal
    m2 = np.zeros(m2_shape, dtype="complex128")
    m2[..., :s1_b, :s1_a] = s1
    m2[..., s1_b:, s1_a:] = s2

    cnx_col = []

    cas_rows = np.arange(s1_b + s2_b)
    # probe row indices should always be located at the bottom of each s-matrix block
    s1_probes = cas_rows[s1_a: s1_b]
    s2_probes = cas_rows[s1_b + s2_a: s1_b + s2_b]
    # external rows indices
    s1_ext = cas_rows[: s1_a]
    s2_ext = cas_rows[s1_b: s1_b + s2_a]
    
    # move columns from m2 to m1 based on the connections
    for i, (p1, p2) in enumerate(connections):
        # if p1 or p2 are greater than the external number of ports, they are internal ports that cannot be 
        # connected.
        if p1 > s1_a or p2 > s2_a:
            raise ValueError(
                f"Cannot connect internal port numbers {p1}, {p2} with shapes: {s1.shape}, {s2.shape}"
            )

        # list of connected columns in the second matrix, these will be zero columns after the connection
        cnx_col += [p1-1, s1_a + p2-1]

        # move connected columns from m2 into m1
        m1[..., :s1_b, s1_b + p2-1] = -m2[..., :s1_b, p1-1]
        m1[..., s1_b:, p1-1] = -m2[..., s1_b:, s1_a + p2-1]

        # if p1 or p2 is a probe port, move the row index of the connected port to the probe list
        if probes[i][0]:
            s1_probes = np.append(s1_probes, s1_ext[p1 - 1])
        if probes[i][1]:
            s2_probes = np.append(s2_probes, s2_ext[p2 - 1])

    # remove the connected row indices from the external port section
    s1_ext = np.delete(s1_ext, connections[:, 0] - 1)
    s2_ext = np.delete(s2_ext, connections[:, 1] - 1)

    # matrix inversion on square matrix m1
    m1_inv = np.linalg.inv(m1)

    # remove the columns of all connected ports
    m2 = np.delete(m2, cnx_col, -1)

    # row indices of the cascaded matrix in the desired order. S1 ports, S2 ports, followed by probe ports.
    row_order = np.concatenate([s1_ext, s2_ext, s1_probes, s2_probes])

    return m1_inv[:, row_order], m2, row_order

    
def connect_py(s1: dict, s2: dict, connections: list, probes: list = None, noise: bool = False):
    """
    Connect multiple ports between two components s1 and s2.
    Connections is of the form [(s1 port, s2 port), (s1 port, s2 port), ...].
    """
    connections = np.atleast_2d(connections)

    if probes is None:
        probes = np.zeros(connections.shape)
    else:
        probes = np.atleast_2d(probes)

    s1_b, s1_a = s1["s"].shape[-2:]
    s2_b, s2_a = s2["s"].shape[-2:]

    b_len = s1_b + s2_b
    a_len = s1_a + s2_a
    m1_shape = (s1["s"].shape[-3], b_len, b_len)
    m2_shape = (s2["s"].shape[-3], b_len, a_len)

    # create first matrix, starts with just the identity matrix
    m1 = np.identity(b_len, dtype="complex128")
    m1 = np.broadcast_to(m1, m1_shape).copy()

    # create second matrix, this is a "block" diagonal matrix with S1, S2 down the diagonal
    m2 = np.zeros(m2_shape, dtype="complex128")
    m2[..., :s1_b, :s1_a] = s1["s"]
    m2[..., s1_b:, s1_a:] = s2["s"]

    cnx_col = []

    cas_rows = np.arange(s1_b + s2_b)
    # probe row indices should always be located at the bottom of each s-matrix block
    s1_probes = cas_rows[s1_a: s1_b]
    s2_probes = cas_rows[s1_b + s2_a: s1_b + s2_b]
    # external rows indices
    s1_ext = cas_rows[: s1_a]
    s2_ext = cas_rows[s1_b: s1_b + s2_a]
    
    # move columns from m2 to m1 based on the connections
    for i, (p1, p2) in enumerate(connections):
        # if p1 or p2 are greater than the external number of ports, they are internal ports that cannot be 
        # connected.
        if p1 > s1_a or p2 > s2_a:
            raise ValueError(
                f"Cannot connect internal port numbers {p1}, {p2} with shapes: {s1.shape}, {s2.shape}"
            )

        # list of connected columns in the second matrix, these will be zero columns after the connection
        cnx_col += [p1-1, s1_a + p2-1]

        # move connected columns from m2 into m1
        m1[..., :s1_b, s1_b + p2-1] = -m2[..., :s1_b, p1-1]
        m1[..., s1_b:, p1-1] = -m2[..., s1_b:, s1_a + p2-1]

        # if p1 or p2 is a probe port, move the row index of the connected port to the probe list
        if probes[i][0]:
            s1_probes = np.append(s1_probes, s1_ext[p1 - 1])
        if probes[i][1]:
            s2_probes = np.append(s2_probes, s2_ext[p2 - 1])

    # remove the connected row indices from the external port section
    s1_ext = np.delete(s1_ext, connections[:, 0] - 1)
    s2_ext = np.delete(s2_ext, connections[:, 1] - 1)

    # matrix inversion on square matrix m1
    m1_inv = np.linalg.inv(m1)

    # remove the columns of all connected ports
    m2 = np.delete(m2, cnx_col, -1)

    # row indices of the cascaded matrix in the desired order. S1 ports, S2 ports, followed by probe ports.
    row_order = np.concatenate([s1_ext, s2_ext, s1_probes, s2_probes])

    cas_data = dict(s=np.einsum("...ij,...jk->...ik", m1_inv[:, row_order], m2, optimize="greedy"))

    if noise:
        # get the columns/rows of the square m1_inv matrix that correspond to external ports (drop probes)
        rows_ext = np.concatenate([np.arange(s1_a), s1_b + np.arange(s2_a)])
        m1_ext = m1_inv[:, rows_ext, :]
        m1_ext = m1_ext[:, :, rows_ext]
        # split columns between the first and second component
        m1_1 = np.array(m1_ext[..., :s1_a], order="C")
        m1_2 = np.array(m1_ext[..., s1_a:], order="C")

        # get the correlation matrices for each component
        c1, c2 = np.array(s1["n"], order="C"), np.array(s2["n"], order="C")

        f_len = s1["s"].shape[0] # number of frequencies
        # run extension function to compute cascaded data, result is written to cas_ndata
        cas_ndata = np.zeros((f_len, a_len, a_len), dtype="complex128")
        core_func.cascade_ndata(m1_1, m1_2, c1, c2, cas_ndata)

        # delete the connected rows/columns
        cas_ndata = np.delete(cas_ndata, cnx_col, axis=-1)
        cas_ndata = np.delete(cas_ndata, cnx_col, axis=-2)

        # save result to output dictionary
        cas_data["n"] = cas_ndata

    return row_order, cas_data

def connect_self_py(s1: dict, p1: int, p2: int, probes: list = None, noise: bool = False):
    """
    Connect two ports between the same component.
    """

    if probes is None:
        probes = np.zeros(2)
    else:
        probes = np.atleast_1d(probes)

    s1_b, s1_a = s1["s"].shape[-2:]
    m1_shape = (s1["s"].shape[-3], s1_b, s1_b)

    # if p1 or p2 are greater than the external number of ports, they are internal probes that cannot be 
    # connected.
    if p1 > s1_a or p2 > s1_a:
        raise ValueError(
            f"Cannot connect internal port numbers {p1}, {p2} with shape: {s1.shape}"
        )

    # create first matrix, starts with just the identity matrix
    m1 = np.identity(s1_b, dtype="complex128")
    m1 = np.broadcast_to(m1, m1_shape).copy()

    # create second matrix
    m2 = np.array(s1["s"])
    # move columns from m2 to m1 based on the connections
    m1[..., p2-1] = m1[..., :, p2-1] - m2[..., :s1_b, p1-1]
    m1[..., p1-1] = m1[..., :, p1-1] - m2[..., :s1_b, p2-1]

    m1_inv = np.linalg.inv(m1)

    # remove the columns of all connected ports
    cnx_col = [p1 - 1, p2 - 1]
    m2 = np.delete(m2, cnx_col, -1)
    
    cas_rows = np.arange(s1_b)
    # probe row indices should always be located at the bottom of each s-matrix block
    s1_ext = cas_rows[: s1_a]
    s1_probes = cas_rows[s1_a: s1_b]

    # if p1 or p2 is a probe port, move the row index of the connected port to the probe list
    if probes[0]:
        s1_probes = np.append(s1_probes, s1_ext[p1 - 1])
    if probes[1]:
        s1_probes = np.append(s1_probes, s1_ext[p2 - 1])

    # delete the connected ports from the external rows
    s1_ext = np.delete(s1_ext, [(p1 - 1), (p2 -1)])

    # append the probe rows to the end of the external rows
    row_order = np.concatenate([s1_ext, s1_probes])

    cas_data = dict(s=np.einsum("...ij,...jk->...ik", m1_inv[:, row_order], m2, optimize="greedy"))

    if noise:
        # get rows corresponding to the external ports
        m1_1 = np.array(m1_inv[:, :s1_a], order="C")
        # get the correlation matrices for the component
        c1 = np.array(s1["n"], order="C")

        f_len = s1["s"].shape[-3] # number of frequencies
        # run extension function to compute cascaded data, result is written to cas_ndata
        cas_ndata = np.zeros((f_len, s1_a, s1_a), dtype="complex128")
        core_func.cascade_self_ndata(m1_1, c1, cas_ndata)

        # remove connected rows/columns
        cas_ndata = np.delete(cas_ndata, [(p1 - 1), (p2 -1)], axis=-1)
        cas_ndata = np.delete(cas_ndata, [(p1 - 1), (p2 -1)], axis=-2)

        # save result to output dictionary
        cas_data["n"] = cas_ndata

    return row_order, cas_data


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