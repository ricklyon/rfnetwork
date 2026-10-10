"""
Uniform Aperture Antenna
============

Generate Far-field pattern for a uniform aperture antenna.
"""

# sphinx_gallery_thumbnail_number = -1

import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv
import pyvista as pv

import rfnetwork as rfn
import mpl_markers as mplm

# set matplotlib style
plt.style.use(rfn.DEFAULT_STYLE)

# %%
# User defined Parameters [inches]
# ------------------------

# solve box size
sbox_h = 7
sbox_w = 7
sbox_len = 7

lam0 = rfn.conv.in_m(rfn.const.c0 / 10e9)

# %%
# Build Model
# -----------

# solve box
sbox = pv.Cube(center=(0, 0, 0), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

# PEC sheet near z=0 for aperture
aperture_z = -sbox_h / 2 + 1
sheet = pv.Rectangle([
    (-sbox_w/2, -sbox_len/2, aperture_z),
    (-sbox_w/2, sbox_len/2, aperture_z),
    (sbox_w/2, sbox_len/2, aperture_z)
])

# cut out aperture from sheet
aperture_w, aperture_len = lam0, 2 * lam0

aperture = pv.Rectangle(
    [(-aperture_w/2, -aperture_len/2, aperture_z), 
    (aperture_w/2, -aperture_len/2, aperture_z),
    (aperture_w/2, aperture_len/2, aperture_z)],
)

sheet_clipped = sheet.clip_box(
    aperture.bounds
).extract_surface(algorithm="dataset_surface")

# add aperture sheet to model
s = rfn.FDTD_Solver(sbox)
s.add_conductor(sheet_clipped, style=dict(color="gold"))

# add absorbing layers on all sides.
s.add_PML("x-", "x+", "y-", "y+", "z-", "z+", n_pml=5)

# use cell size 1/20 of a wavelength
s.generate_mesh(d_max = lam0/20)

# %%
# Add Current Source
# ------------------

# add magnetic current source sheet at aperture surface of 0.1 A/in^2
# e-field is polarized along x axis, so magnetic current is polarized along y.
src = (1 / (conv.m_in(0.1)**2)) * s.gaussian_modulated_source(f0=15e9, width=150e-12, t0=120e-12, t_len=1.0e-9)
s.add_current_source(aperture, "hy", src=src)

s.add_field_monitor("mon1", "ex", "y", position=0, n_step=5)
s.add_farfield_monitor(frequency=10e9)

s.solve(n_threads=4)

fig, ax = plt.subplots()
cpos = pv.CameraPosition(position=(1.5, 2.7, 1.6), focal_point=(0, 0, 0), viewup=(0, 0.0, 1.0))
s.plot_monitor("mon1", init_time=507, camera_position=cpos, axes=ax)
ax.set_title("Uniform Aperture")
fig.tight_layout()

# %%
# Plot Far-field Gain
# ------------------------

theta_cut = s.get_farfield_gain(theta=np.arange(-90, 90, 1), phi=90)

fig, (ax1) = plt.subplots(1, 1, subplot_kw=dict(projection="polar"))
theta_cut.plot("theta", xfmt=np.deg2rad, yfmt="db20", ax=ax1, polarization="phipol")
fig.tight_layout()
ax1.set_xlabel(r"$\theta$ [deg]")
plt.show()
