from np_struct import ldarray
import numpy as np
import pyvista as pv

from rfnetwork import const, core


def translate(pattern: ldarray, position: np.ndarray) -> ldarray:
    """
    Move the phase center of a pattern to a new location.

    Parameters
    ----------
    pattern : ldarray
        Pattern in spherical coordinates. Must have a frequency coordinate in Hz.
    positions : ldarray
        Element positions in meters. Positions can have more than one dimension if provided as a ldarray,
        "axis" dimension must be present with labels "x", "y" and "z".

    Returns
    -------
    pattern
        Labeled array of the pattern data translated to the new location.
        
    """

    # convert pattern coordinates to uvw
    ptn_keys = pattern.coords.keys()

    if "phi" in ptn_keys and "theta" in ptn_keys:
        uvw = uvw_phitheta(pattern.phi, pattern.theta)
    elif "az" in ptn_keys and "el" in ptn_keys:
        uvw = uvw_azel(pattern.az, pattern.el)
    elif "u" in ptn_keys and "v" in ptn_keys:
        uvw = uvw_uv(pattern.u, pattern.v)
    else:
        raise NotImplementedError("Coordinate frame not supported")

    # replace uwv dimension with xyz 
    uvw = ldarray(
        uvw, coords=dict(axis=("x", "y", "z"), **uvw[0].coords)
    )

    # cast as ldarray if not already
    if not isinstance(position, ldarray):
        position = ldarray(position, coords=dict(axis=("x", "y", "z")))

    # dot product (r` r)
    r_dot = np.sum(uvw * position, axis="axis")

    # multiply by the wave number k
    lam = ldarray(const.c0 / pattern.frequency, coords=dict(frequency=pattern.frequency))
    k = 2 * np.pi / lam

    return pattern * np.exp(1j * k * r_dot)

def illuminate_object(
    obj: pv.PolyData, 
    src_pos: tuple, 
    src_patt: ldarray, 
    src_pwr: float,
    frequency: float, 
    spherical: bool = False,
    image: bool = False, 
):
    """
    Illuminate the faces on an object from a source antenna and compute the resulting surface currents.
    
    Parameters
    ----------

    obj: pv.PolyData
        pyvista object with n_cells, cell sizes must be sufficiently small (around 1/10th of wavelength) for good 
        results.
    src_patt: ldarray
        labeled array in rectangular polarization basis of the source antenna. Coordinate frame must be az/el.
    src_pos: tuple
        3-tuple of xyz position of source antenna, in meters
    src_pwr: float
        input power to tx antenna, in watts.
    frequency: float, list
        frequency values to compute surface currents for, in Hz. 
    spherical: bool, default: False
        If True, allows surface currents on the non-illuminated side of a spherical objects. This has negligible effect
        for large objects in the optical region, but approximates the Mie resonances.
    image: bool, default: False
        if True, computes the currents from the ground bounce, assuming the ground is a perfect conductor along the
        xz plane. 

    Returns
    -------
    Js: np.ndarray
        n_cells x 3 array of the electric surface current densities on each face.
    Ms: np.ndarray
        n_cells x 3 array of the magnetic surface current densities on each face.
    r: float
        propagation distance from source to object (including reflection if image is True), in meters
    
    """
    eta0 = const.eta0
    lam0 = const.c0 / frequency
    beta = 2 * np.pi / lam0

    # flip location of source to the other side of ground plane if computing the image
    if image:
        src_pos = list(src_pos)
        src_pos[1] = -src_pos[1]

    # center of each scatterer face, shape (n_cells, 3)
    obj_centers = obj.cell_centers().points
    # unit normal of each scatterer face, pointing out of the surface, shape (n_cells, 3)
    obj_face_normals = obj.cell_normals
    # vector from source location to scatterer
    obj_src_p = obj_centers - src_pos
    # distance r from source to center of each face
    obj_src_r = np.linalg.norm(obj_src_p, axis=-1)
    # normalized vectors from source to each face (direction only)
    poynting_v = (obj_src_p) / obj_src_r[..., None]

    # center of object
    obj_midpoint = np.average(obj_centers, axis=0)
    # maximum distance of any face from midpoint normalized by the wavelength
    obj_max_r = np.max(np.linalg.norm(obj_centers - obj_midpoint, axis=1)) / lam0

    # azimuth angle of each face center from the z axis (antenna boresight), positive towards +x, in degrees
    obj_az = np.rad2deg(np.arctan2(obj_src_p[:, 0], obj_src_p[:, 2]))
    # elevation angle of each face center above the xz plane, positive towards +y, in degrees.
    obj_el = np.rad2deg(np.arctan2(obj_src_p[:, 1], obj_src_p[:, 2]))

    # mirror elevation if image. The pattern flips in elevation because the object is being illuminated by the
    # ground bounce, which starts from the mirrored side of the antenna.
    if image:
        # vertical currents remain in the same direction (y-pol)
        # horizontal currents parallel to ground flip direction in the image
        src_patt[dict(polarization="x")] = -src_patt.sel(polarization="x")
        src_patt[dict(polarization="z")] = -src_patt.sel(polarization="z")
        obj_el = -obj_el

    # evaluate tx pattern at scatterer faces, remove frequency axis. Flattened list of gain values at each objec
    # face
    gain_at_faces = src_patt.sel(frequency=frequency).interpolate(az=obj_az, el=obj_el, flat=True)

    # scale the incident e-field by the effective area of the face that is facing towards the source.
    # cell is facing the source if the dot product between the face normal and poynting vector
    eff_area = np.einsum("ij,ij->i", obj_face_normals, poynting_v)

    if spherical:
        # allow surface currents to wrap around to the non-illuminated side. Coefficients are fit to the Mie series
        # backscatter of a PEC sphere (ka = 0.1 to 60).
        ka = 2 * np.pi * obj_max_r
        # angle past the shadow boundary, 0 at the shadow boundary and pi/2 at the back of the object
        psi = np.arcsin(np.clip(eff_area, 0, 1))
        # low frequency (Rayleigh / first resonance) wrap around, uses the straight through incident phase
        wrap_lf = (1 + 2.046 * ka**0.843) * np.exp(-(ka / 1.268)**4.073)
        # creeping wave launched from the opposite shadow boundary, travels an arc of a(pi - psi) to reach the face.
        # The straight through incident phase (ka sin(psi)) is removed and replaced with the arc length phase.
        creep_amp = (
            (1 - np.exp(-(ka / 1.717)**8.316)) * 0.540 * ka**-2.206 * np.exp((0.846 - 2.008j) * ka**(1 / 3) + 3.864j)
        )
        creep = creep_amp * np.exp(-1j * ka * (np.pi - psi)) * np.exp(1j * ka * np.sin(psi))
        eff_area = np.where(eff_area > 0, (wrap_lf + creep) * eff_area, -eff_area)
    else:
        # allow currents on the non-illuminated side for very small objects, approximates Rayleigh scattering.
        eff_area = np.abs(np.where(eff_area > 0, np.exp(-1.6 * obj_max_r) * eff_area, eff_area))

    # the gain is related to the e-field strength by
    # G = (4pi / 2 eta0) (|rE|^2 / Pin)
    # E = sqrt(Pin G (2 eta0 / 4pi r^2))
    # the tx_patt is already complex voltage gain (sqrt(G))
    # E = g * sqrt(Pin (2 eta0 / 4pi r^2))
    e_inc = gain_at_faces * np.sqrt(src_pwr * (2 * eta0 / (4 * np.pi * obj_src_r**2))) * np.exp(-1j * beta * obj_src_r)
    e_inc = e_inc.transpose(("azel", "polarization")) * eff_area[:, None]

    # cross product with propagation direction vector to get incident h field, H = k^ x E / eta0
    h_inc = np.cross(poynting_v, e_inc) / eta0

    # physical optics electric surface current on the PEC scatterer, Js = n^ x H_inc on lit faces.
    js_at_faces = np.cross(obj_face_normals, h_inc)
    ms_at_faces = np.cross(obj_face_normals, e_inc)

    # distance from source to object center
    r_mid = np.linalg.norm(obj_midpoint - src_pos)

    return js_at_faces, ms_at_faces, r_mid


def get_scattering_rE(
    obj: pv.PolyData, 
    Js: np.ndarray, 
    Ms: np.ndarray, 
    theta: np.ndarray, 
    phi: np.ndarray, 
    frequency: np.ndarray = None, 
    n_threads: int = 4
) -> ldarray:
    """
    Compute far-field scattering pattern for the surface currents on an object.

    E-field values are returned for thetapol and phipol, with the  e^(-jBr) / r term omitted. To compute the true 
    field values at a distance r, multiply the result by e^(-jBr) / r. 

    Parameters
    ----------
    obj: pv.PolyData
        pyvista object with n_cells, cell sizes must be sufficiently small (around 1/10th of wavelength) for good 
        results.
    Js: np.ndarray
        n_cells x 3 array of the electric surface current densities on each face.
    Ms: np.ndarray
        n_cells x 3 array of the magnetic surface current densities on each face.
    theta : np.ndarray | float
        spatial theta values in degrees. These are pair-wise points with phi and must have the same length.
    phi : np.ndarray | float
        spatial phi values in degrees. These are pair-wise points with phi and must have the same length.

    Returns
    -------
    ldarray
        rE field values. labeled numpy array with dimensions (polarization, frequency, phitheta)
    """

    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    frequency = np.atleast_1d(frequency)

    # check that theta and phi are same length
    if theta.ndim != 1 or phi.ndim != 1 or len(theta) != len(phi):
        raise RuntimeError("theta and phi vectors must be equal length.")

    cell_pos = np.array(obj.cell_centers().points.T, order="C", dtype=np.float32)
    # compute pattern centered with the scatterer at the origin of the scatterer
    cell_pos -= np.average(cell_pos, axis=-1)[..., None]

    # area of each face
    sized = obj.compute_cell_sizes(length=False, area=True, volume=False)
    cell_ds = np.array(sized.cell_data["Area"], order="C", dtype=np.float32)

    # max grid length for temporary working grid
    n_cells = obj.n_cells

    ff_data = dict(
        beta = np.array(2 * np.pi * frequency / const.c0, dtype=np.float32, order="C"),
        theta = np.array(np.deg2rad(theta), dtype=np.float32, order="C"),
        phi = np.array(np.deg2rad(phi), dtype=np.float32, order="C"),
        data = np.zeros((2, 1, len(theta)), dtype=np.complex64, order="C"),
        working_grid_cmplx = np.zeros((n_threads, n_cells), dtype=np.complex64, order="C"),
        working_grid_float = np.zeros((n_threads, n_cells), dtype=np.float32, order="C")
    )

    # transpose surface currents so shape is 3 x n_cells
    J_xyz = np.array(Js.T, order="C", dtype=np.complex64)
    M_xyz = np.array(Ms.T, order="C", dtype=np.complex64)

    core.core_func.nf2ff_flat(J_xyz, M_xyz, cell_pos, cell_ds, ff_data, n_threads)

    # cast as labeled array
    return ldarray(
        ff_data["data"],
        coords=dict(polarization=["thetapol", "phipol"], frequency=frequency, phitheta=np.arange(len(theta))),
        attrs=dict(theta=theta, phi=phi)
    )

def get_scattering_rxE(
    obj: pv.PolyData, 
    Js: np.ndarray, 
    Ms: np.ndarray, 
    rx_patt: ldarray,
    rx_pos: np.ndarray, 
    frequency: float
):
    """
    Compute received field at the terminals of an RX antenna, from a scattering object with surface currents Js and Ms.

    Parameters
    ----------
    obj: pv.PolyData
        scattering pyvista object with n_cells, cell sizes must be sufficiently small 
        (around 1/10th of wavelength) for good results.
    Js: np.ndarray
        n_cells x 3 array of the electric surface current densities on each face of scatterer.
    Ms: np.ndarray
        n_cells x 3 array of the magnetic surface current densities on each face of scatterer.
    rx_patt: ldarray
        labeled array in rectangular polarization basis of the source antenna. Coordinate frame must be az/el.
    rx_pos: tuple
        3-tuple of xyz position of source antenna, in meters

    Returns
    -------
    e_field: ldarray
        e-field values. labeled numpy array with dimensions (polarization, frequency). Values are normalized so
        that taking the square will give the power at the RX antenna terminal.
    r: ldarray
        propagation distance in meters from center of scatterer to rx antenna
        
    """

    lam0 = const.c0 / frequency
    beta = 2 * np.pi / lam0

    # midpoint of scatterer
    obj_mid_p = np.average(obj.points, axis=0)
    # vector from center of scatterer to rx antenna
    obj_rx_p = rx_pos - obj_mid_p
    # distance from scatterer to antenna
    obj_rx_r = np.linalg.norm(obj_rx_p, axis=-1)
    # normalized uvw position of rx antenna from the local coordinate frame of scatterer
    obj_rx_v = obj_rx_p / obj_rx_r

    # theta and phi angle back to the rx antenna from scatterer
    phi_rx, theta_rx = phitheta_uvw(*(obj_rx_v.T), flat=True)

    rE_at_rx = get_scattering_rE(obj, Js, Ms, theta_rx, phi_rx, frequency)
    # convert to rectangular polarization
    rE_at_rx = pattern_spherical2rectangular(rE_at_rx).squeeze()

    # propagate rE to get e-field at source
    e_inc_rx = rE_at_rx * np.exp(-1j * beta * obj_rx_r) / obj_rx_r

    # gain of rx pattern at incident direction from scatterer
    rx_az, rx_el = azel_uvw(*(-obj_rx_v.T), flat=True)
    rx_patt_inc = rx_patt.sel(frequency=frequency).interpolate(az=rx_az, el=rx_el, flat=True).squeeze()
    # effective voltage area of rx antenna
    # e_eff = G (lam^2 / 4pi)
    eff_area_rx = np.abs(rx_patt_inc) * np.sqrt(lam0**2 / (4 * np.pi))

    # received field at antenna terminals.
    # e_inc_rx is a peak phasor (same convention as the 2 * eta0 used for the tx field), so the incident
    # power density is |E|^2 / (2 eta0). Divide by sqrt(2 eta0) so the squared result is the received power.
    e_rx_terminal = eff_area_rx * e_inc_rx / np.sqrt(2 * const.eta0)

    return e_rx_terminal, obj_rx_r

def monostatic_rcs(
    obj: pv.PolyData,
    src_pos: tuple,
    src_pwr: float,
    e_rx: np.ndarray,
    frequency: float,
    tx_patt: ldarray,
    rx_patt: ldarray,
):
    """
    Compute monostatic RCS of scatterer given the received e-field values at the RX antenna.

    Parameters
    ----------
    obj: pv.PolyData
        pyvista object with n_cells, cell sizes must be sufficiently small (around 1/10th of wavelength) for good 
        results.
    src_pos: tuple
        3-tuple of xyz position of source antenna, in meters
    src_pwr: float
        input power to tx antenna, in watts.
    e_rx: np.ndarray
        e-field at RX antenna, in rectangular polarization.
    frequency: float, list
        frequency values to compute surface currents for, in Hz. 
    tx_patt: ldarray
        labeled array in rectangular polarization basis of the tx antenna. Coordinate frame must be az/el.
    rx_patt: ldarray
        labeled array in rectangular polarization basis of the rx antenna. Coordinate frame must be az/el.

    Returns
    -------
    rcs: ldarray
        RCS values in rectangular polarization basis
    """
    lam0 = const.c0 / frequency
    
    obj_mid_p = np.average(obj.points, axis=0)
    # solve radar equation for radar cross section
    #  {P_{r}={{P_{t}G_{t}} \over {4\pi r^{2}}}\sigma {{1} \over {4\pi r^{2}}}A_{\mathrm {eff} }}
    # power received
    pwr_rx = np.abs(e_rx)**2

    # azimuth and elevation angle object is at from antenna position
    tx_obj_r = np.linalg.norm(obj_mid_p - src_pos)
    tx_obj_v = obj_mid_p / tx_obj_r
    tx_az, tx_el = azel_uvw(*tx_obj_v, flat=True)
    # tx antenna gain pointing towards object
    tx_gain_v = tx_patt.sel(frequency=frequency).interpolate(az=tx_az, el=tx_el, flat=True).squeeze()

    # gain of rx pattern at incident direction from scatterer
    rx_gain_v = rx_patt.sel(frequency=frequency).interpolate(az=tx_az, el=tx_el, flat=True).squeeze()

    # radar range equation for each polarization
    eff_area_rx = np.abs(rx_gain_v) * np.sqrt(lam0**2 / (4 * np.pi))
    sig_rcs = (pwr_rx / (src_pwr * np.abs(tx_gain_v)**2 * eff_area_rx**2)) * ((4 * np.pi * tx_obj_r**2)**2) 

    return sig_rcs


def _create_meshgrid(*args):
    """ 
    create meshgrid from 1D vectors, pass through if args are already a matrix with more 
    than one dimension.
    """
    if np.all(np.array([a.ndim for a in args]) == 1):
        m_shape = tuple([a.shape[0] for a in args])
        return tuple([np.broadcast_to(a, m_shape) for a in np.ix_(*args)])
    else:
        return args

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
    phi_m, theta_m = _create_meshgrid(phi, theta)

    if deg:
        phi_m, theta_m = np.deg2rad(phi_m), np.deg2rad(theta_m)

    u = np.sin(theta_m) * np.cos(phi_m)
    v = np.sin(theta_m) * np.sin(phi_m)
    w = np.cos(theta_m) * np.ones_like(phi_m)

    # return as labeled arrays, 
    coords = dict(phi=phi, theta=theta)
    return ldarray(u, coords=coords), ldarray(v, coords=coords), ldarray(w, coords=coords)


def phitheta_uvw(u: np.ndarray, v: np.ndarray, w: np.ndarray, deg: bool = True, flat: bool = False):
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
    if not flat:
        u_m, v_m, w_m = _create_meshgrid(u, v, w)
    else:
         u_m, v_m, w_m = u, v, w

    with np.errstate(all='ignore'):
        phi = np.arctan2(v_m, u_m)
        theta = np.arccos(np.clip(w_m, -1, 1))

    if deg:
        phi, theta = np.rad2deg(phi), np.rad2deg(theta)

    # return phi, theta as labeled arrays, with u coordinates in the rows and v coordinates in the columns
    if flat or (isinstance(phi, ldarray) and isinstance(theta, ldarray)):
        return phi, theta
    else:
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
    u_m, v_m = _create_meshgrid(u, v)

    with np.errstate(all='ignore'):
        phi = np.arctan2(v_m, u_m)
        theta = np.arcsin(np.sqrt(u_m**2 + v_m**2))

    if deg:
        phi, theta = np.rad2deg(phi), np.rad2deg(theta)

    # return phi, theta as labeled arrays, with u coordinates in the rows and v coordinates in the columns
    if isinstance(phi, ldarray) and isinstance(theta, ldarray):
        return phi, theta
    else:
        coords = dict(u=u, v=v)
        return ldarray(phi, coords=coords), ldarray(theta, coords=coords)

def uvw_azel(az: np.ndarray, el: np.ndarray, flat: bool = False):
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

    if not flat:
        # form into a meshgrid
        az_m, el_m = _create_meshgrid(az, el)
    else:
        az_m, el_m = az, el

    # convert to degrees
    az_m, el_m = np.deg2rad(az_m), np.deg2rad(el_m)

    u = np.cos(el_m) * np.sin(az_m)
    v = np.sin(el_m)
    w = np.cos(el_m) * np.cos(az_m)

    if flat or (isinstance(az, ldarray) and isinstance(el, ldarray)):
        return az, el
    else:
        coords = dict(az=az, el=el)
        return ldarray(u, coords=coords), ldarray(v, coords=coords), ldarray(w, coords=coords)

def azel_uvw(u: np.ndarray, v: np.ndarray, w: np.ndarray, flat: bool = False):
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
    if not flat:
        u_m, v_m, w_m = _create_meshgrid(u, v, w)
    else:
        u_m, v_m, w_m = u, v, w

    with np.errstate(all='ignore'):
        el = np.arcsin(v_m)
        az = np.arctan2(u_m, w_m)

    # convert to degrees
    az, el = np.rad2deg(az), np.rad2deg(el)

    if flat or (isinstance(az, ldarray) and isinstance(el, ldarray)):
        return az, el
    else:
        coords = dict(u=u, v=v, w=w)
        return  ldarray(az, coords=coords), ldarray(el, coords=coords)

def uvw_uv(u: np.ndarray, v: np.ndarray):
    """
    Convert u, v to u, v, w defined in the upper hemisphere.

    Parameters
    ----------
    u : np.ndarray
        u coordinate
    v : np.ndarray
        v coordinate

    Returns
    -------
    u : ldarray
        u coordinate
    v : ldarray
        v coordinate
    w : ldarray
        w coordinate
    """

    phi, theta = phitheta_uv(u, v)
    u_new, v_new, w_new = uvw_phitheta(phi, theta)

    coords = dict(u=u, v=v)
    return ldarray(u_new, coords=coords), ldarray(v_new, coords=coords), ldarray(w_new, coords=coords)

def pattern_phitheta2azel(pattern: ldarray, az: np.ndarray, el: np.ndarray):
    """
    Convert a far-field pattern from phi, theta coordinates [degrees] to u, v coordinates.
    """
    u, v, w = uvw_azel(az, el)
    phi_i, theta_i = phitheta_uvw(u=u, v=v, w=w)
    return pattern.interpolate(theta=theta_i, phi=phi_i)

def pattern_uv2azel(pattern: ldarray, az: np.ndarray, el: np.ndarray):
    """
    Convert a far-field pattern from phi, theta coordinates [degrees] to u, v coordinates.
    """
    u, v, w = uvw_azel(az, el)
    return pattern.interpolate(u=u, v=v)

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


def pattern_spherical2rectangular(pattern: ldarray):
    r"""
    Convert pattern defined with spherical polarization vectors to rectangular polarization.
    """

    if not all(pattern.polarization == ("thetapol", "phipol")):
        raise ValueError("Pattern polarization must be defined with spherical component vectors.")

    if "phitheta" in pattern.coords.keys():
        theta_m, phi_m = pattern.theta, pattern.phi 
    else:
        pattern = np.transpose(pattern, (..., "theta", "phi"))
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

    # use "t" for spatial axis if phi and theta are flattened into a single dimension, otherwise use "tp"
    tp_axis = "t" if "phitheta" in pattern.coords.keys() else "tp"
    # matrix multiply by the transformation matrix to get rectangular polarization vectors
    return ldarray(
        np.einsum(f"nm{tp_axis},m...{tp_axis}->n...{tp_axis}", A, pattern), coords=coords
    )


def pattern_rectangular2spherical(pattern: ldarray):
    """
    Project pattern from cartesian polarization vectors to spherical polarization.

    """
    if not all(pattern.polarization == ("x", "y", "z")):
        raise ValueError("Pattern polarization must be defined with cartesian component vectors.")

    pattern = np.transpose(pattern, (..., "theta", "phi"))

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
        np.einsum("nmtp,m...tp->n...tp", A, pattern), coords=coords
    )

def pattern_spherical2cp(pattern: ldarray):
    """
    Project pattern from spherical polarization to circular polarization
    """
    if not all(pattern.polarization == ("thetapol", "phipol")):
        raise ValueError("Pattern polarization must be defined with spherical component vectors.")

    pattern = np.transpose(pattern, (..., "theta", "phi"))
    
    # project phi/theta to specified polarization. project vectors project from thetapol, phipol to 
    # rhcp, lhcp
    A = [
        [1 / np.sqrt(2), 1j / np.sqrt(2)],  # advance phi component by 90 deg for rhcp
        [1 / np.sqrt(2), -1j / np.sqrt(2)]  # delay phi component by 90 deg for lhcp
    ]

    coords = dict(**pattern.coords)
    coords["polarization"] = ["rhcp", "lhcp"]

    # matrix multiply by the transformation matrix to get spherical polarization vectors
    return ldarray(
        np.einsum("nm,m...tp->n...tp", A, pattern), coords=coords
    )

def pattern_spherical2azel(pattern: ldarray):
    """
    Project pattern from spherical polarization to Ludwig's II definition.
    """
    if not all(pattern.polarization == ("thetapol", "phipol")):
        raise ValueError("Pattern polarization must be defined with spherical component vectors.")

    pattern = np.transpose(pattern, (..., "theta", "phi"))
    
    theta_m, phi_m = np.meshgrid(pattern.theta, pattern.phi, indexing="ij")

    # convert to radians
    theta_m, phi_m = np.deg2rad(theta_m), np.deg2rad(phi_m)

    # rows are az, el; columns are thetapol, phipol
    A = np.array(
        [[np.cos(phi_m), -np.cos(theta_m) * np.sin(phi_m)], 
         [np.cos(theta_m) * np.sin(phi_m), np.cos(phi_m)]],
    )
    with np.errstate(all='ignore'):
        cosEl = np.sqrt(1 - np.sin(theta_m) ** 2 * np.sin(phi_m) ** 2)
        A *= np.where(cosEl == 0, 0, 1 / cosEl)

    coords = dict(**pattern.coords)
    coords["polarization"] = ["l2_az", "l2_el"]

    # matrix multiply by the transformation matrix to get new polarization vectors
    return ldarray(
        np.einsum("nmtp,m...tp->n...tp", A, pattern), coords=coords
    )

def pattern_azel2spherical(pattern: ldarray):
    """
    Project pattern from Ludwig's II definition to spherical polarization.
    """
    if not all(pattern.polarization == ("l2_az", "l2_el")):
        raise ValueError("Pattern polarization must be defined with spherical component vectors.")

    pattern = np.transpose(pattern, (..., "theta", "phi"))

    theta_m, phi_m = np.meshgrid(pattern.theta, pattern.phi, indexing="ij")

    # convert to radians
    theta_m, phi_m = np.deg2rad(theta_m), np.deg2rad(phi_m)
    
    # rows are az, el; columns are thetapol, phipol
    A = np.array(
        [[np.cos(phi_m), -np.cos(theta_m) * np.sin(phi_m)], 
         [np.cos(theta_m) * np.sin(phi_m), np.cos(phi_m)]],
    )
    with np.errstate(all='ignore'):
        cosEl = np.sqrt(1 - np.sin(theta_m) ** 2 * np.sin(phi_m) ** 2)
        A *= np.where(cosEl == 0, 0, 1 / cosEl)

    # take inverse of A to reverse the projection. Matrix is orthogonal.
    A_inv = np.transpose(A, (1, 0, 2, 3))

    coords = dict(**pattern.coords)
    coords["polarization"] = ["thetapol", "phipol"]

    # matrix multiply by the transformation matrix to get new polarization vectors
    return ldarray(
        np.einsum("nmtp,m...tp->n...tp", A_inv, pattern), coords=coords
    )