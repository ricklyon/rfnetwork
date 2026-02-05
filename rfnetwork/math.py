import numpy as np
from np_struct import ldarray
from . component import Component_Data, Component
from . core.units import const, conv

def shift_reference_plane(component : Component, distance: float, port: int = 1, er: float = 1) -> Component:
    """
    Move the s-parameter reference plane of a s-parameter Component.

    Parameters
    ----------
    component : Component
        s-parameter component to modify
    distance : float, list
        distance in inches to move the reference plane [inches]. Positive values move the reference plane away from the
        component port (adds phase delay). Negative values move the plane towards the port.
    port : int, list
        port number(s) of the component to modify the reference plane for. A list of ports is supported if each value
        matches with the list of the d argument.
    er : float, default: 1
        relative permittivity of the transmission line that the reference plane is moved through.

    Returns
    -------
    Component
        new component with the new reference plane.
    
    """
    # cast distance and port as lists
    distance = np.atleast_1d(distance)
    port = np.atleast_1d(port)

    # get s-parameter matrix from component
    sdata = component.evaluate(noise=False)["s"]
    # wave number of transmission line at port
    frequency = sdata.coords["frequency"]
    beta = 2 * np.pi * frequency / (const.c0 / np.sqrt(er))

    # build a full array of distances for each port. Distance argument may be incomplete or out of order.
    N = sdata.shape[-1]
    d_meters = np.zeros(N)
    for (d, p) in zip(distance, port):
        d_meters[p-1] = conv.m_in(d)

    # build diagonal matrix of electrical distances at each port that the reference plane needs to move by
    bl = beta[..., None] * d_meters[None]
    phs = np.zeros_like(sdata)
    phs[:, np.arange(N), np.arange(N)] = np.exp(-1j * bl)

    # move the reference plane
    s_prime = np.linalg.matmul(phs, np.linalg.matmul(sdata, phs))

    return Component_Data(
        ldarray(s_prime, coords=sdata.coords)
    )