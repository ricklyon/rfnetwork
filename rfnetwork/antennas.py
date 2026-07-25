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


def phitheta_uvw(u: np.ndarray, v: np.ndarray, w: np.ndarray, deg: bool = True):
    """
    Convert far-field spatial coordinates from (u, v) to (phi, theta)

    Parameters
    ----------
    u : np.ndarray | float
        u coordinates
    v : np.ndarray | float
        v coordinates
    w : np.ndarray | float
        w coordinates
    deg : bool, default = True
        If True, returned coordinates are in degrees, radians otherwise.

    Returns
    -------
    phi, theta : np.ndarray
        Tuple of 2 values, each is a 2D meshgrid of phi, theta angles at each u, v coordinate
    """
    u, v, w = np.atleast_1d(u), np.atleast_1d(v), np.atleast_1d(w)
    # form into a meshgrid
    u_m, v_m = np.meshgrid(u, v, indexing="ij")

    with np.errstate(all='ignore'):
        phi = np.arctan2(v_m, u_m)
        theta = np.arccos(np.clip(w, -1, 1))

    if deg:
        phi, theta = np.rad2deg(phi), np.rad2deg(theta)

    # return phi, theta as labeled arrays, with u coordinates in the rows and v coordinates in the columns
    coords = dict(u=u, v=v, w=w)
    return ldarray(phi, coords=coords), ldarray(theta, coords=coords)


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


def uvw_azel(az: np.ndarray, el: np.ndarray):
    """
    Convert azimuth and elevation coordinates [degrees] to u, v, w.

    Parameters
    ----------
    az : np.ndarray
        azimuth coordinates
    el : np.ndarray
        elevation coordinates

    Returns
    -------
    u : ldarray
        u coordinate
    v : ldarray
        v coordinate
    w : ldarray
        w coordinate
    """
    az, el = np.atleast_1d(az), np.atleast_1d(el)
    # form into a meshgrid
    az_m, el_m = np.meshgrid(az, el, indexing="ij")

    # convert to degrees
    az_m, el_m = np.deg2rad(az_m), np.deg2rad(el_m)

    u = np.cos(el_m) * np.sin(az_m)
    v = np.sin(el_m)
    w = np.cos(el_m) * np.cos(az_m)

    coords = dict(az=az, el=el)
    return ldarray(u, coords=coords), ldarray(v, coords=coords), ldarray(w, coords=coords)

def azel_uvw(u: np.ndarray, v: np.ndarray, w: np.ndarray):
    """
    Convert u, v, w to azimuth and elevation coordinates [degrees].

    Parameters
    ----------
    u : np.ndarray
        u coordinate
    v : np.ndarray
        v coordinate
    w : np.ndarray
        w coordinate

    Returns
    -------
    az : ldarray
        azimuth coordinate
    el : ldarray
        elevation coordinate

    """
    u, v, w = np.atleast_1d(u), np.atleast_1d(v), np.atleast_1d(w)
    # form into a meshgrid
    u_m, v_m, w_m = np.meshgrid(u, v, w, indexing="ij")

    with np.errstate(all='ignore'):
        el = np.arcsin(v_m)
        az = np.arctan2(u_m, w_m)

    # convert to degrees
    az, el = np.rad2deg(az), np.rad2deg(el)

    coords = dict(u=u, v=v, w=w)
    return ldarray(az, coords=coords), ldarray(el, coords=coords)

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


def pattern_project2rectangular(pattern: ldarray):
    r"""
    Convert pattern defined with spherical polarization vectors to rectangular polarization.
    """

    if not all(pattern.polarization == ("thetapol", "phipol")):
        raise ValueError("Pattern polarization must be defined with spherical component vectors.")
    
    theta_m, phi_m = np.meshgrid(pattern.theta, pattern.phi, indexing="ij")

    # convert to radians
    theta_m, phi_m = np.deg2rad(theta_m), np.deg2rad(phi_m)

    # transformation matrix that converts from A_theta, A_phi component vectors (along the columns)
    # to Ax, Ay, Az (rows).
    A = np.array(
        [
            [np.cos(theta_m) * np.cos(phi_m), -np.sin(phi_m)],
            [np.cos(theta_m) * np.sin(phi_m), np.cos(phi_m)],
            [-np.sin(theta_m), np.zeros_like(phi_m)],
        ],
    )

    coords = dict(**pattern.coords)
    coords["polarization"] = ["x", "y", "z"]

    # matrix multiply by the transformation matrix to get rectangular polarization vectors
    return ldarray(
        np.einsum("nmtp,mftp->nftp", A, pattern), coords=coords
    )


def pattern_project2spherical(pattern: ldarray):
    r"""
    Convert pattern defined with cartesian polarization vectors to spherical polarization.

    """
    if not all(pattern.polarization == ("x", "y", "z")):
        raise ValueError("Pattern polarization must be defined with cartesian component vectors.")
    
    theta_m, phi_m = np.meshgrid(pattern.theta, pattern.phi, indexing="ij")

    # convert to radians
    theta_m, phi_m = np.deg2rad(theta_m), np.deg2rad(phi_m)

    # transformation matrix that converts from Ax, Ay, Az component vectors (along the columns) 
    # to A_theta, A_phi (rows).
    A = np.array(
        [
            [np.cos(theta_m) * np.cos(phi_m), np.cos(theta_m) * np.sin(phi_m), -np.sin(theta_m)],
            [-np.sin(phi_m), np.cos(phi_m), np.zeros_like(phi_m)],
        ],
    )

    coords = dict(**pattern.coords)
    coords["polarization"] = ["thetapol", "phipol"]

    # matrix multiply by the transformation matrix to get spherical polarization vectors
    return ldarray(
        np.einsum("nmtp,mftp->nftp", A, pattern), coords=coords
    )
