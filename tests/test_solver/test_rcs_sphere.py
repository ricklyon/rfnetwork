import pyvista as pv

from np_struct import ldarray
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rfnetwork as rfn

from pathlib import Path

from rfnetwork import const, conv, antennas

plt.style.use(rfn.DEFAULT_STYLE)

dir_ = Path(__file__).parent

eta0 = 377


def illuminate_object(
    obj: pv.PolyData, 
    src_pos: tuple, 
    src_patt: ldarray, 
    src_pwr: float,
    frequency: float, 
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
    image: bool, default: False
        if True, computes the currents from the ground bounce, assuming the ground is a perfect conductor along the
        xz plane. 

    Returns
    -------
    Js: np.ndarray
        n_cells x 3 array of the electric surface current densities on each face.
    Ms: np.ndarray
        n_cells x 3 array of the magnetic surface current densities on each face.
    
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
        src_patt = -src_patt
        obj_el = -obj_el

    # evaluate tx pattern at scatterer faces, remove frequency axis. Flattened list of gain values at each objec
    # face
    gain_at_faces = src_patt.sel(frequency=frequency).interpolate(az=obj_az, el=obj_el, flat=True)

    # scale the incident e-field by the effective area of the face that is facing towards the source.
    # cell is facing the source if the dot product between the face normal and poynting vector
    eff_area = np.einsum("ij,ij->i", obj_face_normals, poynting_v)
    # allow surface currents to wrap around to the non-illuminated side for small objects
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

    return js_at_faces, ms_at_faces

obj_max_r = np.logspace(-1, 1)

# plt.plot(obj_max_r, np.exp(-obj_max_r - 1))
# plt.xscale("log")
# plt.yscale("log")
# plt.ylim([0.001, 1])

##############################
# Build Scene
##############################
frequency = 3e9
lam0 = rfn.const.c0 / frequency
beta = 2 * np.pi / lam0

pv.set_jupyter_backend("trame")

# isotropic source antenna at the origin
az, el = np.arange(-90, 92, 2), np.arange(-90, 92, 2)
tx_patt = ldarray(
    np.zeros((3, 1, len(az), len(el))),
    coords=dict(polarization=("x", "y", "z"), frequency=frequency, az=az, el=el)
)
tx_patt[dict(polarization="y")] = 1

src_patt = tx_patt

scatterer1_loc = (0, 0, 500)
src_pwr = 1
rx_pos = np.array((0, 0, 0))
src_pos = np.array((0, 0, 0))
rx_patt = tx_patt.copy()

# build scene, dimensions in meters
# z points out from antenna, x is azimuth, y is elevation (antenna centered coordinate frame)
# # scatterer1 = pv.read(pv.examples.planefile)
# scatterer1 = pv.Rectangle(
#     ((-1, -1, 0), (-1, 1, 0), (1, 1, 0))
# )
# scatterer1 = scatterer1.triangulate().subdivide(7, subfilter='linear')

# scatterer1 = pv.Box((-1, 1, -1, 1, -0.05, 0.05))

#  rotate to desired orientation with center at origin
# rot = Rotation.from_euler("xyz", (-90, 70, 20), degrees=True).as_matrix()
# scatterer1.points = np.einsum("ij, ...j->...i", rot, scatterer1.points)


# sphere radius in wavelengths to test rcs
r_lam = np.logspace(-2, 1, 101)
src_pos = (0, 0, 0)
src_patt = tx_patt

rcs_actual = np.zeros_like(r_lam)

r = r_lam[0] 

for i, r in enumerate(r_lam):
    # reset sphere to unit size
    res = np.clip(int(2 * np.pi * r * 10), 60, 1000)
    obj = pv.Sphere(radius=r * lam0, theta_resolution=res, phi_resolution=res)

    # move to desired location
    obj.points += scatterer1_loc

    Js, Ms = illuminate_object(obj, src_pos, src_patt, src_pwr, frequency)

    # plotter = pv.Plotter()
    # scatterer1.cell_data["Ms (db)"] = rfn.conv.db20_lin(np.linalg.norm(Ms, axis=-1)) 
    # scatterer1.cell_data["Ms (ang)"] = np.angle(Ms[:, 0], deg=True)

    # scatterer1.cell_data["Js (db)"] = rfn.conv.db20_lin(np.linalg.norm(Js, axis=-1)) 
    # scatterer1.cell_data["Js (ang)"] = np.angle(Js[:, 1], deg=True)


    # plotter.add_mesh(scatterer1, cmap="jet", scalars="Js (db)", clim=(-110, -93))
    # # add_polar_lines(plotter, Js, Ms, frequency)
    # plotter.add_axes()

    # plotter.show()

    # theta = np.arange(-180, 182, 2)
    # rE_local_phi0 = antennas.get_scattering_rE(scatterer1, Js, Ms, theta, np.ones_like(theta) * 0, frequency).squeeze()
    # rE_local_phi0.coords["phitheta"] = theta
    # rE_local_phi0 = rE_local_phi0.rename(phitheta="theta")

    # fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    # rE_local_phi0.plot("theta", xfmt=np.deg2rad, yfmt="db20", polarization="phipol")
    # plt.show()

    ##############################
    # Propagate back to RX antennas
    ##############################


    # vector from center of scatterer to rx antenna
    obj_rx_p = rx_pos - scatterer1_loc
    # distance from scatterer to antenna
    obj_rx_r = np.linalg.norm(obj_rx_p)
    # normalized uvw position of rx antenna from the local coordinate frame of scatterer
    obj_rx_v = obj_rx_p / obj_rx_r

    # theta and phi angle back to the rx antenna from scatterer
    phi_rx, theta_rx = rfn.antennas.phitheta_uvw(*(obj_rx_v.T), flat=True)

    rE_at_rx = antennas.get_scattering_rE(obj, Js, Ms, theta_rx, phi_rx, frequency)
    # convert to rectangular polarization
    rE_at_rx = rfn.antennas.pattern_spherical2rectangular(rE_at_rx).squeeze()

    # propagate rE to get e-field at source
    e_inc_rx = rE_at_rx * np.exp(-1j * beta * obj_rx_r) / obj_rx_r

    # gain of rx pattern at incident direction from scatterer
    rx_az, rx_el = rfn.antennas.azel_uvw(*(-obj_rx_v.T), flat=True)
    rx_patt_inc = rx_patt.sel(frequency=frequency).interpolate(az=rx_az, el=rx_el, flat=True).squeeze()
    # effective voltage area of rx antenna
    # e_eff = G (lam^2 / 4pi)

    eff_area_rx = np.abs(rx_patt_inc) * np.sqrt(lam0**2 / (4 * np.pi))
    # received field at antenna terminals.
    # e_inc_rx is a peak phasor (same convention as the 2 * eta0 used for the tx field), so the incident
    # power density is |E|^2 / (2 eta0). Divide by sqrt(2 eta0) so the squared result is the received power.
    e_rx_terminal = eff_area_rx * e_inc_rx / np.sqrt(2 * eta0)

    ################
    # solve radar equation for radar cross section
    ################
    #  {P_{r}={{P_{t}G_{t}} \over {4\pi r^{2}}}\sigma {{1} \over {4\pi r^{2}}}A_{\mathrm {eff} }}

    # power received
    pwr_rx = np.abs(e_rx_terminal)**2

    # tx antenna gain pointing towards object
    tx_obj_p = scatterer1_loc
    tx_obj_r = np.linalg.norm(tx_obj_p)
    tx_obj_v = tx_obj_p / tx_obj_r
    tx_az, tx_el = rfn.antennas.azel_uvw(*tx_obj_v, flat=True)
    tx_gain_v = tx_patt.interpolate(az=tx_az, el=tx_el).sel(frequency=frequency).squeeze()

    # radar range equation for each polarization, first rx antenna
    sig_rcs = (pwr_rx / (src_pwr * np.abs(tx_gain_v)**2 * eff_area_rx**2)) * ((4 * np.pi * tx_obj_r**2)**2) 

    rcs_actual[i] = sig_rcs.sel(polarization="y")

    # analytical rcs for a sphere
    # rcs_expected = np.pi * (sph_rad * lam0)**2
    # analytical rcs for a flat plate at normal incidence, sigma = 4 pi A^2 / lam^2
    # plate_area = (scatterer1_radius * 2)**2
    # rcs_expected = ((4 * np.pi) / lam0**2) * plate_area**2

    # print("expected rcs", rfn.conv.db10_lin(rcs_expected))
    # print("actual rcs", rfn.conv.db10_lin(sig_rcs.sel(polarization="y")))

    # print(rfn.conv.db10_lin(sig_rcs.sel(polarization="y") / rcs_expected))


    # plot analytical pattern from page 341 in Banalis

# plot normalized rcs
proj_area = np.pi * (r_lam * lam0)**2
ka = 2 * np.pi * r_lam 


fig, ax = plt.subplots()
ax.plot(ka, rcs_actual / proj_area)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([0.001, 10])
ax.set_xlim([0.1, 10])