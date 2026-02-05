import numpy as np
from np_struct import ldarray
from . core.units import const, conv

def shift_reference_plane(sdata : ldarray, distance: list, er: list) -> ldarray:
    """
    Move the s-parameter reference plane of a s-parameter Component.

    Parameters
    ----------
    component : ldarray
        s-parameter data to modify
    distance : list
        distance in inches to move the reference plane [meters] for each port. Positive values move the reference 
        plane away from the component port (adds phase delay). Negative values move the plane towards the port.
    er : list
        relative permittivity of the transmission line that the reference plane is moved through, at each port.

    Returns
    -------
    ldarray
        new s-matrix with the new reference plane.
    
    """
    # cast distance and port as lists
    distance = np.atleast_1d(distance)
    er = np.atleast_1d(er)
    N = len(distance)

    # wave number of transmission line at port
    frequency = sdata.coords["frequency"]
    # beta is size FxN
    beta = 2 * np.pi * frequency[..., None] / (const.c0 / np.sqrt(er)[None])

    # build diagonal matrix of electrical distances at each port that the reference plane needs to move by
    bl = beta * distance[None]
    phs = np.zeros_like(sdata)
    phs[:, np.arange(N), np.arange(N)] = np.exp(-1j * bl)

    # move the reference plane
    s_prime = np.linalg.matmul(phs, np.linalg.matmul(sdata, phs))

    return ldarray(s_prime, coords=sdata.coords)