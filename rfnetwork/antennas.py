from np_struct import ldarray
import numpy as np

def uvw_phitheta(phi: np.ndarray, theta: np.ndarray, deg: bool = True):
    """
    Convert far-field spatial coordinates from (phi, theta) to (u, v, w).

    Parameters
    ----------
    phi : np.ndarray | float
        phi coordinates
    theta : np.ndarray | float
        theta coordinates
    deg : bool, default = True
        If True, phi and theta coordinates are interpreted in degrees, radians otherwise.

    Returns
    -------
    u, v, w : np.ndarray
        Tuple of three values, each is a 2D meshgrid of u, v, w values at each phi, theta angle. 
    """
    phi, theta = np.atleast_1d(phi), np.atleast_1d(theta)
    # form into a meshgrid
    phi_m, theta_m = np.meshgrid(phi, theta, indexing="ij")

    if deg:
        phi_m, theta_m = np.deg2rad(phi_m), np.deg2rad(theta_m)

    u = np.sin(theta_m) * np.cos(phi_m)
    v = np.sin(theta_m) * np.sin(phi_m)
    w = np.cos(theta_m) * np.ones_like(phi_m)

    # return as labeled arrays, 
    coords = dict(phi=phi, theta=theta)
    return ldarray(u, coords=coords), ldarray(v, coords=coords), ldarray(w, coords=coords)


def phitheta_uv(u: np.ndarray, v: np.ndarray, deg: bool = True):
    """
    Convert far-field spatial coordinates from (u, v) to (phi, theta)

    Parameters
    ----------
    u : np.ndarray | float
        u coordinates
    v : np.ndarray | float
        v coordinates
    deg : bool, default = True
        If True, returned coordinates are in degrees, radians otherwise.

    Returns
    -------
    phi, theta : np.ndarray
        Tuple of 2 values, each is a 2D meshgrid of phi, theta angles at each u, v coordinate
    """
    u, v = np.atleast_1d(u), np.atleast_1d(v)
    # form into a meshgrid
    u_m, v_m = np.meshgrid(u, v, indexing="ij")

    with np.errstate(all='ignore'):
        phi = np.arctan2(v_m, u_m)
        theta = np.arcsin(np.sqrt(u_m**2 + v_m**2))

    if deg:
        phi, theta = np.rad2deg(phi), np.rad2deg(theta)

    # return phi, theta as labeled arrays, with u coordinates in the rows and v coordinates in the columns
    coords = dict(u=u, v=v)
    return ldarray(phi, coords=coords), ldarray(theta, coords=coords)


def pattern_phitheta2uv(pattern: ldarray, u: np.ndarray, v: np.ndarray):
    """
    Convert a far-field pattern from phi, theta coordinates [degrees] to u, v coordinates.
    """
    phi_i, theta_i = phitheta_uv(u=u, v=v)
    return pattern.interpolate(theta=theta_i, phi=phi_i)


def pattern_uv2phitheta(pattern: ldarray, phi: np.ndarray, theta: np.ndarray):
    """
    Convert a far-field pattern from u, v coordinates to phi, theta coordinates [degrees].
    """
    u_i, v_i, _ = uvw_phitheta(phi=phi, theta=theta)
    return pattern.interpolate(u=u_i, v=v_i)
