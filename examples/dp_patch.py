"""
Dual-Polarized Patch Antenna
============

Build a dual polarized patch antenna based on [1].

[1] Meltem Yildirim, "Design of Dual Polarized Wideband Microstrip Antennas", pp. 54-70.
"""

# sphinx_gallery_thumbnail_number = -3

from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 

import pyvista as pv
from np_struct import ldarray

import rfnetwork as rfn
from rfnetwork import conv
import mpl_markers as mplm
import sys

np.set_printoptions(suppress=True)
# set matplotlib style
plt.style.use(rfn.DEFAULT_STYLE)

pv.set_jupyter_backend("trame")
sys.argv = sys.argv[0:1]

try:
    dir_ = Path(__file__).parent
except:
    dir_ = Path().cwd()

# %%
# User defined Parameters [inches]
# ------------------------

f0 = 2.21e9
lam0 = rfn.const.c0_in / f0

# bottom substrate er
er_btm = 2.54
h_btm = conv.in_mm(1.6)
# top substrate er
er_top = 2.54
h_top = conv.in_mm(1.6)

# parameters from table 3-1 in [1]
len_patch = conv.in_mm(30)
w_patch = conv.in_mm(40)
w_slot = conv.in_mm(1.55)
len_slot = conv.in_mm(6)
len_leg = conv.in_mm(4)
w_ms = conv.in_mm(4.42)
len_stub = conv.in_mm(20)


# %%
# Build Model
# ------------------------

# solve box
sbox = pv.Cube(center=(0, 0, 0), x_length=w_patch*1.7, y_length=w_patch*1.7, z_length=lam0 / 5)

# create model and add elements
s = rfn.FDTD_Solver(sbox)

# top substrate
sub_x0, sub_x1, sub_y0, sub_y1 = (-w_patch * 0.6, w_patch * 0.6, -len_patch * 0.6, len_patch * 0.6)
sub_top = pv.Box(bounds=(sub_x0, sub_x1, sub_y0, sub_y1, 0, h_top))
sub_btm = pv.Box(bounds=(sub_x0, sub_x1, sub_y0, sub_y1, -h_btm, 0))
s.add_dielectric(sub_top, er=er_top, style=dict(opacity=0.2))
s.add_dielectric(sub_btm, er=er_btm, style=dict(opacity=0.2))

# center conductor layer with slot
gnd_plane = pv.Rectangle([(sub_x0, sub_y0, 0), (sub_x0, sub_y1, 0), (sub_x1, sub_y1, 0)])
# cutout slots
gnd_plane = gnd_plane.clip_box((-w_slot/2, w_slot/2, -len_slot/2, len_slot/2, 0, 0)).extract_surface(algorithm="dataset_surface")
gnd_plane = gnd_plane.clip_box((-len_leg/2, len_leg/2, -len_slot/2 - w_slot/2, -len_slot/2 + w_slot/2, 0, 0)).extract_surface(algorithm="dataset_surface")
gnd_plane = gnd_plane.clip_box((-len_leg/2, len_leg/2, len_slot/2 - w_slot/2, len_slot/2 + w_slot/2, 0, 0)).extract_surface(algorithm="dataset_surface")
s.add_conductor(gnd_plane, style=dict(color="k"))


# create patch
patch = pv.Rectangle(
    [(-w_patch/2, -len_patch/2, h_top), (-w_patch/2, len_patch/2, h_top), (w_patch/2, len_patch/2, h_top)]
)
s.add_conductor(patch, style=dict(opacity=0.4))

# microstrip feed trace
port_x = sub_x0 + 0.1
ms_trace = pv.Rectangle(
    [(port_x, -w_ms/2, -h_btm), (port_x, w_ms/2, -h_btm), (len_stub, w_ms/2, -h_btm)]
)
s.add_conductor(ms_trace)

# add port
port1_face = pv.Rectangle([(port_x, -w_ms/2, 0), (port_x, w_ms/2, 0), (port_x, w_ms/2, -h_btm)])
s.add_lumped_port(1, port1_face, "z-")

# PML boundaries
s.assign_PML_boundaries("x-", "x+", "y-", "y+", "z+", "z-", n_pml=5)

s.generate_mesh(d0 = 0.05, d_edge=0.025)
s.render().show()


s.plot_coefficients("ex_z", "b", "z", 0).show()

# setup far-field monitor
s.add_farfield_monitor(frequency=f0)

s.add_field_monitor("mon1", "ez", "z", h_top, n_step=10)

# # show model rendering
# cpos = pv.CameraPosition(
#     position=(25, 25, 10),
#     focal_point=(5, 0, 0),
#     viewup=(0, 0.0, 1.0),
# )

# fig, ax = plt.subplots()
# plotter = s.render(show_mesh=False, show_rulers=False, axes=ax, camera_position=cpos)

# %%
# Setup Excitation and Solve
# ------------------------
vsrc = s.gaussian_source(width=120e-12, t0=60e-12, t_len=20000e-12)
# plt.plot(vsrc)

s.assign_excitation(vsrc, 1)
s.solve(n_threads=4)

s.plot_monitor("mon1", opacity=1).show()


# plot far-field cut along theta at phi=0
theta_cut = rfn.conv.db10_lin(
    s.get_farfield_gain(theta=np.arange(-180, 181, 2), phi=0).sel(polarization="thetapol")
)

fig1, ax = plt.subplots(subplot_kw=dict(projection="polar"))
theta_rad = np.deg2rad(theta_cut.coords["theta"])
ax.plot(theta_rad, theta_cut.squeeze().T)

ax.set_theta_zero_location('N') 
ax.set_theta_direction(-1) 
ax.set_ylim([-25, 10])
ax.set_yticks(np.arange(-25, 15, 5))
ax.set_yticklabels(["", "-20", "-15", "-10", "-5", "0", "5", "10dBi"])

# Set theta labels
ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
labels = [f"{d}°" for d in [0, 45, 90, 135, 180, -135, -90, -45]]
ax.set_xticklabels(labels)

ax.set_xlabel(r"$\theta$ [deg], $\phi$=0°")
ax.legend(["{:.3f}GHz".format(f/1e9) for f in theta_cut.coords["frequency"]])
mplm.line_marker(x=0)
plt.show()

# %%
# Plot S11
# ------------------------

frequency: np.ndarray = np.arange(1.5e9, 3e9, 2e6)
sdata = s.get_sparameters(frequency, downsample=False)

S11 = ldarray(sdata[:, 0][..., None, None], coords=dict(frequency=sdata.coords["frequency"], b=[1], a=[1]))

ant = rfn.Component_Data(S11)

fig, ax = plt.subplots()
ant.plot(11, fmt="smith", axes=ax)


fig, ax = plt.subplots()
ant.plot(11, fmt="db", axes=ax)


mplm.line_marker(x=f0 / 1e9, xlabel=True)
plt.show()
# %%
