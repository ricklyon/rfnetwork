"""
PML Terminated Micro-strip
============

Terminate a microstrip line with a PML layer and evaluate PML attenuation.
"""

# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import unittest

import rfnetwork as rfn
import mpl_markers as mplm
from pathlib import Path
import numpy as np

# set matplotlib style
plt.style.use(rfn.DEFAULT_STYLE)

try:
    dir_ = Path(__file__).parent
except:
    dir_ = Path().cwd()

# %%
# User defined Parameters [inches]
# ------------------------

# axis parallel to microstrip line
len_axis = "z"
# axis along width of microstrip
width_axis = "x"
# axis perpendicular to microstrip
normal_axis = "y"

# width and length 
ms_w = 0.04
ms_len = 2

# solve bos size
sbox_h = 1
sbox_w = 1.5
sbox_len = ms_len * 1.5

sub_h = 0.02
ms_ends = (-0.4, sbox_len/2)

f0 = 10e9

# %%
# Build Model
# -----------

# axis indices
la, wa, na = [("x", "y", "z").index(ax) for ax in (len_axis, width_axis, normal_axis)]

line_ref = rfn.elements.MSLine(h=sub_h, er=3.66, w=ms_w, length=ms_len * 1.0)
z_ref = line_ref.get_properties(f0).sel(value="z0").item()

def build_dims(len, width, height):
    dimensions = [None for i in range(3)]

    dimensions[la] = len
    dimensions[wa] = width
    dimensions[na] = height

    return dimensions

sub_size = build_dims(sbox_len, sbox_w, sub_h)
substrate = pv.Cube(
    center=build_dims(0, 0, sub_h/2), 
    x_length=sub_size[0], y_length=sub_size[1], z_length=sub_size[2]
)

sbox_size = build_dims(sbox_len, sbox_w, sbox_h)
sbox = pv.Cube(
    center=build_dims(0, 0, sbox_h/2), 
    x_length=sbox_size[0], y_length=sbox_size[1], z_length=sbox_size[2]
)

ms1_trace = pv.Rectangle([
    build_dims(ms_ends[0], - ms_w/2, sub_h),
    build_dims(ms_ends[0], + ms_w/2, sub_h),
    build_dims(ms_ends[1], + ms_w/2, sub_h)
])

port1_face = pv.Rectangle([
    build_dims(ms_ends[0], - ms_w/2, sub_h),
    build_dims(ms_ends[0], + ms_w/2, sub_h),
    build_dims(ms_ends[0], + ms_w/2, 0),
])

current_face = pv.Rectangle([
    build_dims(0, - ms_w/2 - 0.001, sub_h + 0.001),
    build_dims(0, + ms_w/2 + 0.001, sub_h + 0.001),
    build_dims(0, + ms_w/2 + 0.001, sub_h - 0.001),
])


voltage_line = pv.Line(
    build_dims(0, 0, sub_h), build_dims(0, 0, 0)
)

s = rfn.FDTD_Solver(sbox)
s.add_dielectric(substrate, er=3.66, style=dict(opacity=0.0))
s.add_conductor(ms1_trace, style=dict(color="gold"))

int_axis = ["x+", "y+", "z+"][na]
s.add_lumped_port(1, port1_face, integration_line=int_axis)

# add PML
pml_axis = ["x-", "x+", "y-", "y+", "z+", "z-"]
pml_axis.remove(f"{normal_axis}-")
s.add_PML(*pml_axis, n_pml=10)
# s.add_PML(f"{len_axis}+", f"{len_axis}-", n_pml=5)

# edge correction
p1 = build_dims(ms_ends[0], + ms_w/2, sub_h)
p2 = build_dims(ms_ends[1], + ms_w/2, sub_h)

s.edge_correction(p1, p2, f"{width_axis}+")

p1 = build_dims(ms_ends[0], - ms_w/2, sub_h)
p2 = build_dims(ms_ends[1], - ms_w/2, sub_h)
s.edge_correction(p1, p2, f"{width_axis}-")

s.generate_mesh(d_max = 0.02)

# %%
# Solve Model
# -----------

# efield normal to trace
e_normal = f"e{int_axis[0]}"
s.add_field_monitor("mon1", e_normal, e_normal[1], sub_h, 5)

s.add_current_probe("c1", current_face)
s.add_voltage_probe("v1", voltage_line)

vsrc = 1 * s.gaussian_source(width=120e-12, t0=80e-12, t_len=800e-12)
frequency: np.ndarray = np.arange(f0 - 2e9, f0+2e9, 10e6)

s.assign_excitation(vsrc, 1)
s.solve(n_threads=4, show_progress=False)

# %%
# Plot PML Attenuation
# -----------
# Show the amount of attenuation from th PML layer

#  .. image:: ../_static/img/msline_pml.gif

# get the current and voltage values from each probe
line_i = s.vi_probe_values("c1")
line_v = s.vi_probe_values("v1")

# plot time domain voltage from probe
fig, ax = plt.subplots()
ax.plot(s.time / 1e-9, conv.db20_lin(vsrc), label="Applied Voltage", alpha=0.5)
ax.plot(s.time / 1e-9, conv.db20_lin(line_i * 50), label="Probe Current * Z0")
ax.legend()
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Power [dB]")
ax.set_ylim([-120, 0])
ax.grid(True)

cpos = pv.CameraPosition(position=(1, 1, 1), focal_point=(0, 0, 0), viewup=(0, 1, 0))
gif_setup = dict(file = dir_ / "../docs/_static/img/msline_pml.gif", fps=15, step_ps=10)
p = s.plot_monitor(["mon1"], vmax=60, vmin=0, gif_setup=gif_setup, camera_position=cpos, opacity=0.9)

# %%
# Plot Line Impedance
# -------------------
sdata = s.get_sparameters(frequency, downsample=False)
S11 = sdata[:, 0]

# compute line impedance
IP = utils.dtft(s.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(s.vi_probe_values("v1"), frequency, 1 / s.dt)
ZP = VP / IP

fig, ax = plt.subplots()
ax.plot(frequency / 1e9, ZP.real)
ax.plot(frequency / 1e9, conv.z_gamma(S11))
ax.set_ylim([0, 120])
ax.axhline(y=z_ref, linestyle=":", color="k")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
mplm.line_marker(x = f0 / 1e9, axes=ax)
ax.legend(["Probe", "Port"])
ax.set_title(f"Width Axis: {wa}, Length Axis: {la}, Normal Axis: {na}")

fig.tight_layout()
plt.show()
