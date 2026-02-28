"""
Coupled Lines
============

Even and odd mode impedance of coupled microstrip lines.
"""

import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import rfnetwork as rfn
import mpl_markers as mplm

frequency = np.arange(5e9, 15e9, 10e6)

# %%
# Design Parameters 
# ------------------------
#
ms_w = 0.03  # trace width
ms_sp = 0.005  # trace spacing
sub_h = 0.03  # substrate height
er =  3.66  # relative permittivity

# solve box dimensions
sbox_h = 0.25
sbox_w = 0.3
sbox_len = 0.25

# center locations of microstrip lines along y axis
ms1_y = -(ms_w / 2) - (ms_sp / 2)
ms2_y = (ms_w / 2) + (ms_sp / 2)

# end locations of lines along x axis, lines terminate in PML region 
ms_x = (-sbox_len/2 + 0.1, sbox_len/2)

# substrate geometry
substrate = pv.Cube(
    center=(0, 0, sub_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sub_h
)

# solve box
sbox = pv.Cube(
    center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h
)

# %%
# Create 3D Model
# ------------------------
#
s = rfn.FDTD_Solver(sbox)
s.add_dielectric(substrate, er=er, style=dict(opacity=0.2))

# add microstrip lines
for i, ms_y in enumerate((ms1_y, ms2_y)):
    ms_trace = pv.Rectangle([
        (ms_x[0], ms_y - ms_w/2, sub_h),
        (ms_x[0], ms_y + ms_w/2, sub_h),
        (ms_x[1], ms_y + ms_w/2, sub_h)
    ])

    port_face = pv.Rectangle([
        (ms_x[0], ms_y - ms_w/2, sub_h),
        (ms_x[0], ms_y + ms_w/2, sub_h),
        (ms_x[0], ms_y + ms_w/2, 0),
    ])

    s.add_conductor(ms_trace, style=dict(color="gold"))
    s.add_lumped_port(i+1, port_face, integration_axis="z-")

# assign PML layers, omitting the x- side near the ports
s.assign_PML_boundaries("x+", "z+", "y-", "y+", n_pml=5)

# create mesh with a nominal width of 20mils far from geometry edges, and 2.5mils near edges.
# cell widths are tapered to minimize errors
s.generate_mesh(d0 = 0.02, d_edge = 0.0025)

# apply edge singularity correction to the edges down the length of the microstrip lines
for i, ms_y in enumerate((ms1_y, ms2_y)):
    p1 = (ms_x[0], ms_y + ms_w/2, sub_h)
    p2 = (ms_x[1], ms_y + ms_w/2, sub_h)

    s.edge_correction(p1, p2, integration_axis="y+")

    p1 = (ms_x[0], ms_y - ms_w/2, sub_h)
    p2 = (ms_x[1], ms_y - ms_w/2, sub_h)

    s.edge_correction(p1, p2, integration_axis="y-")

# add 2D field monitor normal to the x-axis at the center of the grid
s.add_field_monitor("mon1", "e_total", axis="x", position=0)

# %%
# Setup Excitations
# ------------------------
#

# create voltage waveform
vsrc = 1e-2 * s.gaussian_modulated_source(f0=10e9, width=280e-12, t0=160e-12, t_len=400e-12)

fig, ax = plt.subplots()
ax.plot(vsrc.coords["time"] * 1e12, vsrc)
ax.set_xlabel("Time [ps]")
ax.set_ylabel("Voltage")

# %%
# Run Even Mode 
# ------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# set up camera to view the fields near the ports, looking down the x axis
cpos = pv.CameraPosition(
    position=(-0.15, -0.05, 0.1),
    focal_point=(0, 0, 0.03),
    viewup=(0, 0.0, 1.0),
)

# arguments for plot_monitor
plot_mon_kwargs = dict(
    monitor=["mon1", "mon1"], 
    style=["vectors", "surface"],  # plot both a vector field and magnitude colormap
    max_vector_len=0.005,  # keep vectors shorter than 5mils
    opacity="linear",   # fade smaller field components
    init_time=165,  # start at 165ps
    show_mesh=False,
    show_rulers=False,
    camera_position=cpos,
    vmax=35,  # maximum colormap value, in dB because linear is False by default
)

# run even mode, same waveform at both port 1 and 2
s.assign_excitation(vsrc, [1, 2])
s.solve(n_threads=4, show_progress=False)
S_even = s.get_sparameters(frequency)

s.plot_monitor(**plot_mon_kwargs, axes=ax1)

# %%
# Run Odd Mode 
# ------------------------

# setup opposite polarity waveforms at each port
s.assign_excitation(vsrc, 1)
s.assign_excitation(-vsrc, 2)
s.solve(n_threads=4, show_progress=False)
S_odd = s.get_sparameters(frequency)

s.plot_monitor(**plot_mon_kwargs, axes=ax2)

ax1.set_title("Even Mode")
ax2.set_title("Odd Mode")

fig.tight_layout(pad=0.5)


# %%
# Plot Coupled Line Impedance
# ------------------------

# reference even and odd impedance are taken from this online solver:
# https://wcalc.sourceforge.net/cgi-bin/coupled_microstrip.cgi

ref_even_z = 101.847
ref_odd_z = 45.0888

fig, ax = plt.subplots()
ax.plot(frequency / 1e9, conv.z_gamma(S_odd.sel(b=1, a=1)))
ax.plot(frequency / 1e9, conv.z_gamma(S_even.sel(b=1, a=1)))

plt.ylim([0, 110])
plt.axhline(y=ref_odd_z, linestyle=":", color="tab:blue")
plt.axhline(y=ref_even_z, linestyle=":", color="tab:orange")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
ax.legend(["Even Mode", "Odd Mode", "Ref Odd", "Ref Even"])
mplm.line_marker(x = 10, axes=ax)
