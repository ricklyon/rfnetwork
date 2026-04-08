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

import rfnetwork as rfn
from rfnetwork import conv
import mpl_markers as mplm
import sys

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

f0 = 2.2e9
lam0 = rfn.const.c0_in / f0

# bottom substrate er
er_btm = 2.54
h_btm = conv.in_mm(1.6)
# top substrate er
er_top = 2.54
h_top = conv.in_mm(1.6)

# parameters from table 3-1 in [1]
len_patch = conv.in_mm(40)
w_patch = conv.in_mm(30)
w_slot = conv.in_mm(1.55)
len_slot = conv.in_mm(11.2)
len_ms = conv.in_mm(58)
w_ms = conv.in_mm(4.42)
len_stub = conv.in_mm(20)


# %%
# Build Model
# ------------------------

# solve box
sbox = pv.Cube(center=(0, 0, 0), x_length=w_patch*1.7, y_length=len_patch*1.7, z_length=lam0 / 4)

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
slot_cutout = pv.Box((-w_slot/2, w_slot, -len_slot/2, len_slot/2, -h_btm, h_top))
gnd_plane = gnd_plane.clip_box((-w_slot/2, w_slot, -len_slot/2, len_slot/2, 0, 0)).extract_surface(algorithm="dataset_surface")
s.add_conductor(gnd_plane, style=dict(opacity=0.5, color="black"))

# gnd_plane.plot()

# create patch
patch = pv.Rectangle(
    [(-w_patch/2, -len_patch/2, h_top), (-w_patch/2, len_patch/2, h_top), (w_patch/2, len_patch/2, h_top)]
)
s.add_conductor(patch, style=dict(opacity=0.4))

# point_cloud = pv.PolyData(gnd_plane.points)
# plotter = pv.Plotter()
# plotter.add_mesh(point_cloud, color="k")
# plotter.add_mesh(gnd_plane, opacity=0.5)
# plotter.show()

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


self = s
s.generate_mesh(d0 = 0.05, d_edge=0.015)
s.render().show()


# s.plot_coefficients("ex_z", "b", "z", 0).show()

# setup far-field monitor
# s.add_farfield_monitor(frequency=f0)

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
vsrc = s.gaussian_source(width=100e-12, t0=100e-12, t_len=5000e-12)
# plt.plot(vsrc)

s.assign_excitation(vsrc, 1)
s.solve(n_threads=4)

s.plot_monitor("mon1", opacity=1).show()

# # %%
# # Principal Plane Cut at phi=0°
# # ------------------------
# # This plot shows realized gain

# phi_cut = rfn.conv.db10_lin(
#     s.get_farfield_gain(phi=np.arange(-180, 182, 2), theta=90).sel(polarization="thetapol")
# )

# theta_cut = rfn.conv.db10_lin(
#     s.get_farfield_gain(theta=np.arange(-180, 181, 2), phi=0).sel(polarization="thetapol")
# )

# fig1, ax1 = plt.subplots(subplot_kw=dict(projection="polar"))
# fig2, ax2 = plt.subplots(subplot_kw=dict(projection="polar"))

# theta_rad = np.deg2rad(theta_cut.coords["theta"])
# phi_rad = np.deg2rad(phi_cut.coords["phi"])

# ax1.plot(theta_rad, theta_cut.squeeze())
# ax2.plot(phi_rad, phi_cut.squeeze())

# for ax in (ax1, ax2):
#     ax.set_theta_zero_location('N') 
#     ax.set_theta_direction(-1) 
#     ax.set_ylim([-20, 10])
#     ax.set_yticks(np.arange(-20, 15, 5))
#     ax.set_yticklabels(["", "-15", "-10", "-5", "0", "5", "10dBi"])

#     # Set theta labels
#     ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
#     labels = [f"{d}°" for d in [0, 45, 90, 135, 180, -135, -90, -45]]
#     ax.set_xticklabels(labels)

# ax1.set_xlabel(r"$\theta$ [deg], $\phi$=0°")
# ax2.set_xlabel(r"$\phi$ [deg], $\theta$=90°")
# mplm.line_marker(x=np.pi/2, axes=ax1, xline=False)

# fig.tight_layout()

# %%
# Plot S11
# ------------------------
frequency: np.ndarray = np.arange(500e6, 5e9, 10e6)
sdata = s.get_sparameters(frequency, downsample=False)
S11 = sdata[:, 0]

fig, ax = plt.subplots()
ax.plot(frequency / 1e6, conv.db20_lin(S11))
ax.set_ylim([-20, 5])
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("[dB]")

mplm.line_marker(x=f0 / 1e6, xlabel=True)
plt.show()