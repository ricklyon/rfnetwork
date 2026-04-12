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

# todo:
# 1. make patch cp
# 2. implement rhcp, lhcp, xyz polarizations
# 3. add angled msline tests
# 4. parallel threads for far-field calculation
# 5. ldarray interpolation
# 6. make function to combine s-data into a single matrix


try:
    dir_ = Path(__file__).parent
except:
    dir_ = Path().cwd()

# %%
# User defined Parameters [inches]
# ------------------------

f0 = 2.4e9
lam0 = rfn.const.c0_in / f0

# bottom substrate er
er_btm = 3.66
h_btm = 0.04
# top substrate er
er_top = 3.66
h_top = 0.06

# parameters from table 3-1 in [1]
len_patch = conv.in_mm(30)
w_patch = conv.in_mm(30)
w_slot = conv.in_mm(1.55)
len_slot = conv.in_mm(7.5)
len_leg = conv.in_mm(4.5)
w_ms = 0.09
len_stub1 = 0.15#0.6
len_stub2 = 0.55

# upper slot offset (h-polarized patch)
x_pos_h = conv.in_mm(-9.1)
# lower (v-polarized patch)
y_pos_v = conv.in_mm(-9.1)

rfn.elements.MSLine(0.04, 3.5, z0=50)



# %%
# Build Model
# ------------------------

# solve box
sbox = pv.Cube(center=(0, 0, 0), x_length=w_patch*1.8, y_length=w_patch*1.8, z_length=lam0 / 5)

# create model and add elements
s = rfn.FDTD_Solver(sbox)

# top substrate
sub_x0, sub_x1, sub_y0, sub_y1 = (-w_patch * 0.8, w_patch * 0.8, -len_patch * 0.8, len_patch * 0.8)
sub_top = pv.Box(bounds=(sub_x0, sub_x1, sub_y0, sub_y1, 0, h_top))
sub_btm = pv.Box(bounds=(sub_x0, sub_x1, sub_y0, sub_y1, -h_btm, 0))
s.add_dielectric(sub_top, er=er_top, style=dict(opacity=0.2))
s.add_dielectric(sub_btm, er=er_btm, style=dict(opacity=0.2))

# center conductor layer with slot
gnd_plane = pv.Rectangle([(sub_x0, sub_y0, 0), (sub_x0, sub_y1, 0), (sub_x1, sub_y1, 0)])
# cutout slots
slot_h = (x_pos_h-w_slot/2, x_pos_h+  w_slot/2, -len_slot/2, len_slot/2, 0, 0)
leg1_h = (x_pos_h-len_leg/2, x_pos_h+len_leg/2, -len_slot/2 - w_slot/2, -len_slot/2 + w_slot/2, 0, 0)
leg2_h = (x_pos_h-len_leg/2, x_pos_h+len_leg/2, len_slot/2 - w_slot/2, len_slot/2 + w_slot/2, 0, 0)

# # cutout slot for lower (V)
slot_v = (-len_slot/2, len_slot/2, y_pos_v - w_slot/2, y_pos_v + w_slot/2, 0, 0)
leg1_v = (-len_slot/2 - w_slot/2, -len_slot/2 + w_slot/2, y_pos_v - len_leg/2, y_pos_v + len_leg/2, 0, 0)
leg2_v = (len_slot/2 - w_slot/2, len_slot/2 + w_slot/2, y_pos_v - len_leg/2, y_pos_v + len_leg/2, 0, 0)

for cutout in (slot_h, leg1_h, leg2_h, slot_v, leg1_v, leg2_v):
    gnd_plane = gnd_plane.clip_box(cutout).extract_surface(algorithm="dataset_surface")

s.add_conductor(gnd_plane, style=dict(color="k", opacity=0.8))


# create patch
patch = pv.Rectangle(
    [(-w_patch/2, -len_patch/2, h_top), (-w_patch/2, len_patch/2, h_top), (w_patch/2, len_patch/2, h_top)]
)
s.add_conductor(patch, style=dict(opacity=0.4))

# microstrip feed trace for H pol slot
port_x = sub_x0 + 0.05
ms_trace_h = pv.Rectangle(
    [(port_x, -w_ms/2, -h_btm), (port_x, w_ms/2, -h_btm), (x_pos_h+len_stub1, w_ms/2, -h_btm)]
)
ms_stub_h = pv.Rectangle(
    [(x_pos_h+len_stub1 - w_ms/2, 0, -h_btm), 
     (x_pos_h+len_stub1 + w_ms/2, 0, -h_btm), 
     (x_pos_h+len_stub1 + w_ms/2, len_stub2, -h_btm)]
)
s.add_conductor(ms_trace_h, ms_stub_h)

# microstrip feed for V pol slot
port_y = sub_y0 + 0.05
ms_trace_v = pv.Rectangle(
    [(-w_ms/2, port_y, -h_btm), (w_ms/2, port_y, -h_btm), (w_ms/2, y_pos_v + len_stub1, -h_btm)]
)
ms_stub_v = pv.Rectangle(
    [(0, y_pos_v + len_stub1 - w_ms/2, -h_btm), 
     (0, y_pos_v + len_stub1 + w_ms/2, -h_btm), 
     (len_stub2, y_pos_v + len_stub1 + w_ms/2, -h_btm)]
)
s.add_conductor(ms_trace_v, ms_stub_v)


# add port
port1_face = pv.Rectangle([(port_x, -w_ms/2, 0), (port_x, w_ms/2, 0), (port_x, w_ms/2, -h_btm)])
s.add_lumped_port(1, port1_face, "z-")

port2_face = pv.Rectangle([(-w_ms/2, port_y, 0), (w_ms/2, port_y, 0), (w_ms/2, port_y, -h_btm)])
s.add_lumped_port(2, port2_face, "z-")

# PML boundaries
s.assign_PML_boundaries("x-", "x+", "y-", "y+", "z+", "z-", n_pml=5)

s.generate_mesh(d0 = 0.08, d_edge=0.02)
s.render().show()


# s.plot_coefficients("ex_z", "b", "z", 0).show()

# setup far-field monitor
s.add_farfield_monitor(frequency=f0)

s.add_field_monitor("mon1", "ez", "z", 0, n_step=50)

# # show model rendering
# cpos = pv.CameraPosition(
#     position=(25, 25, 10),
#     focal_point=(5, 0, 0),
#     viewup=(0, 0.0, 1.0),
# )

# fig, ax = plt.subplots()
# plotter = s.render(show_mesh=False, show_rulers=False, axes=ax, camera_position=cpos)

# %%
# Animated Gif
# ------------------------
# vsrc1 = s.gaussian_modulated_source(f0, width=5000e-12, t0=2500e-12, t_len=5000e-12)


# # apply a phase delay to the vertically polarized port to get RHCP
# wt0 = np.pi / 2
# t_delay = wt0 / (2 * np.pi * f0)
# # number of steps that fit in the delay (rounded, no interpolation)
# n_delay = int(np.around(t_delay / s.dt))
# vsrc2 = np.roll(vsrc1, n_delay)

# plt.plot(vsrc1)
# plt.plot(vsrc2)

# s.assign_excitation(vsrc1, 1)
# s.assign_excitation(vsrc2, 2)

# s.solve(n_threads=4)

# gif_setup = dict(file = dir_ / "dp_patch.gif", fps=20, step_ps=20, stop_ps=2000)
# s.plot_monitor("mon1", opacity="linear", gif_setup=gif_setup)


# %%
# Setup Excitation and Solve
# ------------------------
vsrc1 = s.gaussian_modulated_source(f0, width=100e-12, t0=60e-12, t_len=21000e-12)


# apply a phase delay to the vertically polarized port to get RHCP
wt0 = np.pi / 2
t_delay = wt0 / (2 * np.pi * f0)
# number of steps that fit in the delay (rounded, no interpolation)
n_delay = int(np.around(t_delay / s.dt))
vsrc2 = np.roll(vsrc1, n_delay)

plt.figure()
plt.plot(vsrc1)
plt.plot(vsrc2)

s.assign_excitation(vsrc1, 1)
s.assign_excitation(vsrc2, 2)

s.solve(n_threads=4)

# plot far-field cut along theta at phi=0
theta_cut = rfn.conv.db10_lin(
    s.get_farfield_gain(theta=np.arange(-180, 181, 2), phi=0, polarization=["rhcp", "lhcp"])
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
ax.legend(["RHCP", "LHCP"])
mplm.line_marker(x=0)
plt.show()

# %%
# Plot S11
# ------------------------

frequency: np.ndarray = np.arange(1.5e9, 3e9, 2e6)
sdata = s.get_sparameters(frequency, downsample=False)

ant = rfn.Component_Data(sdata)

fig, ax = plt.subplots()
ant.plot(11, fmt="smith", axes=ax)


fig, ax = plt.subplots()
ant.plot(11, fmt="db", axes=ax)


mplm.line_marker(x=f0 / 1e9, xlabel=True)


fig, ax = plt.subplots()
ant.plot(21, fmt="db", axes=ax)
# %%
plt.show()


