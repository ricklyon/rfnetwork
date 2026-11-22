"""
Conducting Sphere RCS
=====================

Compute the monostatic RCS of a conducting sphere.
"""

# sphinx_gallery_thumbnail_number = 3

import pyvista as pv
import vtk
from np_struct import ldarray
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rfnetwork as rfn
from rfnetwork import antennas

from pathlib import Path

dir_ = Path(__file__).parent

# set matplotlib style
plt.style.use(rfn.DEFAULT_STYLE)
pv.set_jupyter_backend("trame")

# %%
# User defined Parameters [meters]
# ------------------------

frequency = 3e9
lam0 = rfn.const.c0 / frequency
# radius of sphere in meters
radius_list = np.array([0.02, 0.08, 0.25])

# isotropic source antenna at the origin
# TX beam pattern, y-polarization
src_patt = ldarray.load(dir_ / "data/vivaldi_antenna_patt.npy")
tx_pin = 1 # input power in watts

# xyz location of sphere in meters
sphere_loc = (0, 0, 2)
# tx power in watts
src_pwr = 1
# position of receiving and tx antenna
rx_pos = np.array((0, 0, 0))
src_pos = np.array((0, 0, 0))
rx_patt = src_patt.copy()

# %%
#  Build Models
# ------------
rcs = [0] * len(radius_list)
rE = [None] * len(radius_list)

for i, radius in enumerate(radius_list):
    # set theta/phi resolution of sphere object to around 1/5 of a wavelength
    res = np.clip(int(2 * np.pi * radius / lam0 * 5), 60, 500)
    obj = pv.Sphere(radius=radius, theta_resolution=res, phi_resolution=res)

    # move to desired location
    obj.points += sphere_loc

    # compute surface currents on sphere from transmitter
    Js, Ms, _ = antennas.illuminate_object(obj, src_pos, src_patt, src_pwr, frequency, spherical=True)
    # get scattered fields back at the rx antenna
    e_rx_terminal, _ = antennas.get_scattering_rxE(obj, Js, Ms, rx_patt, rx_pos, frequency)
    # compute rcs of sphere in the y-polarization (co-pol)
    rcs[i] = antennas.monostatic_rcs(
        obj, src_pos, src_pwr, e_rx_terminal, frequency, src_patt, rx_patt
    ).sel(polarization="y")

    # %%
    #  Plot Scene
    # ------------
    plotter = pv.Plotter()
    obj.cell_data["Js"] = np.linalg.norm(Js, axis=-1) 
    obj.cell_data["Ms"] = np.linalg.norm(Ms, axis=-1) 

    # add antenna rendering, scale to roughly 5x3in
    antenna = pv.read(dir_ / "data/horn_antenna.stl")
    antenna.points *= (rfn.conv.m_in(5) / np.max(antenna.points[:, 0]))
    # move to desired location
    antenna.points += src_pos

    plotter.add_mesh(antenna, color="tan")
    plotter.add_mesh(obj, cmap="jet", scalars="Ms")
    plotter.add_axes()
    plotter.show_bounds(bounds=(-0.5, 0.5, -0.5, 0.5, 0, 2), show_zlabels=False, show_xlabels=False, show_ylabels=False, xtitle=" ", ytitle=" ", ztitle=" ")
    plotter.add_bounding_box()

    cpos = pv.CameraPosition(position=(-1.9900, 1.0525, -2.1575), focal_point=(-0.1086, -0.0931, 0.7756), viewup=(0, 1, 0))
    plotter.camera_position = cpos
    plotter.show()

    # get scattering pattern
    theta = np.arange(-180, 182, 2)
    phi = np.zeros_like(theta)

    rE_local = rfn.antennas.get_scattering_rE(obj, Js, Ms, theta, phi, frequency)
    rE_local = rE_local.rename(phitheta="theta")
    rE_local.coords["theta"] = theta
    rE[i] = rE_local

# %%
# Plot Scattering Pattern
# -----------------------
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
for rE_i in rE:
    rE_i.plot("theta", xfmt=np.deg2rad, yfmt="db20", polarization="phipol")
plt.show()

# %%
# Compare with Analytical Values
# ------------------------------

r_lam = np.logspace(-2, 1, 241)
ref_rcs = np.array([rfn.utils.sphere_mie_monostatic(r * lam0, frequency) for r in r_lam])

fig, ax = plt.subplots()
ax.plot(2 * np.pi * r_lam, ref_rcs / (np.pi * (r_lam * lam0)**2), alpha=0.3, color="k")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(f"$2\pi r / \lambda$")
ax.set_ylabel("RCS / $\pi r^2$")

ax.plot([2 * np.pi * radius_list / lam0], [rcs / (np.pi * (radius_list)**2)], marker=".")
