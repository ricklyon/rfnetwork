"""
Yagi Antenna
============

Simulate UHF Yagi antenna and plot far-field gain.
"""

import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv
import pyvista as pv
import sys

import rfnetwork as rfn
import mpl_markers as mplm

pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)
sys.argv = sys.argv[0:1]


# %%
# User defined Parameters [inches]
# ------------------------
# sphinx_gallery_thumbnail_number = -1

f0 = 440e6
lam0 = rfn.const.c0_in / f0

# wire diameter
wire_d = 0.2
# gap between driver elements
gap = 0.2

driver_len = 0.89 * (lam0 / 2)

# reflector_len = lam0 * 0.482
# director1_len = lam0 * 0.428
# director2_len = lam0 * 0.424

reflector_len = lam0 * 0.49
director1_len = lam0 * 0.428
director2_len = lam0 * 0.416

sp = 0.22 * lam0

# %%
# Build Yagi Model
# ------------------------

driver_upper = pv.Cylinder(
    center=(0, 0, gap / 2 + driver_len / 4), direction=(0, 0, 1), radius = wire_d / 2, height = driver_len / 2, resolution=4
)

driver_lower = pv.Cylinder(
    center=(0, 0, -gap / 2 - driver_len / 4), direction=(0, 0, 1), radius = wire_d / 2, height = driver_len / 2, resolution=4
)

reflector_upper = pv.Cylinder(
    center=(-sp, 0, reflector_len / 4), direction=(0, 0, 1), radius = wire_d / 2, height = reflector_len / 2, resolution=4
)

reflector_lower = pv.Cylinder(
    center=(-sp, 0, - reflector_len / 4), direction=(0, 0, 1), radius = wire_d / 2, height = reflector_len / 2, resolution=4
)

director1_upper = pv.Cylinder(
    center=(sp, 0, director1_len / 4), direction=(0, 0, 1), radius = wire_d / 2, height = director1_len / 2, resolution=4
)

director1_lower = pv.Cylinder(
    center=(sp, 0, - director1_len / 4), direction=(0, 0, 1), radius = wire_d / 2, height = director1_len / 2, resolution=4
)

director2_upper = pv.Cylinder(
    center=(sp * 2, 0, director2_len / 4), direction=(0, 0, 1), radius = wire_d / 2, height = director2_len / 2, resolution=4
)

director2_lower = pv.Cylinder(
    center=(sp * 2, 0, - director2_len / 4), direction=(0, 0, 1), radius = wire_d / 2, height = director2_len / 2, resolution=4
)

# solve box
sbox = pv.Cube(center=(sp/2, 0, 0), x_length=lam0 * 1.1, y_length=lam0 / 2, z_length=lam0)

# port between upper and lower leg
port1_face = pv.Rectangle([
    (0, -wire_d/2, gap/2),
    (0, wire_d/2, gap/2),
    (0, wire_d/2, -gap/2)
])

s = rfn.FDTD_Solver(sbox)

# add elements
s.add_conductor(
    driver_upper, driver_lower, 
    reflector_upper, reflector_lower,
    director1_lower, director1_upper,
    director2_lower, director2_upper,
    style=dict(color="gold", opacity=0.9))
s.add_lumped_port(1, port1_face, "z-")

# PML boundaries are required on all sides to add a far-field monitor
s.assign_PML_boundaries("x-", "x+", "y-", "y+", "z+", "z-", n_pml=5)
s.generate_mesh(d0 = 0.5, d_edge=0.1)

# setup wide-band far-field monitor
s.add_farfield_monitor(frequency=f0)

# near-field monitor
s.add_field_monitor("e_tot", "e_total", "y", 0, n_step=10)


plotter = s.render(show_mesh=True)
plotter.show()

# %%
# Setup Excitation and Solve
# ------------------------
vsrc = s.gaussian_source(width=800e-12, t0=500e-12, t_len=30e-9)

# plt.plot(vsrc)
s.assign_excitation(vsrc, 1)
s.solve(n_threads=4)

s.plot_monitor(["e_tot"], zoom=1.1, opacity=1, camera_position="xz").show()


# %%
# Principal Plane Cut at phi=0°
# ------------------------
# This plot shows realized gain

pp_gain = rfn.conv.db10_lin(
    s.get_farfield_gain(theta=np.arange(-180, 181, 1), phi=[0]).sel(polarization="thetapol")
)

fig, (ax) = plt.subplots(1, 1, subplot_kw=dict(projection="polar"))
theta_rad = np.deg2rad(pp_gain.coords["theta"])

ax.plot(theta_rad, pp_gain.squeeze())

ax.set_theta_zero_location('N') 
ax.set_theta_direction(-1) 
ax.set_xlabel(r"$\theta$ [deg], $\phi$=0°")
ax.set_ylim([-20, 10])
ax.set_yticks(np.arange(-20, 15, 5))
ax.set_yticklabels(["", "-15", "-10", "-5", "0", "5", "10dBi"])
ax.legend(loc="lower right")
mplm.line_marker(x=np.pi/2)

# Set theta labels
ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
labels = [f"{d}°" for d in [0, 45, 90, 135, 180, -135, -90, -45]]
ax.set_xticklabels(labels)

fig.tight_layout()

# %%
# Plot S11
# ------------------------
frequency: np.ndarray = np.arange(1e6, 1e9, 1e6)
sdata = s.get_sparameters(frequency, downsample=False)
S11 = sdata[:, 0]

fig, ax = plt.subplots()
ax.plot(frequency / 1e6, conv.db20_lin(S11))
ax.set_ylim([-20, 5])
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("[dB]")
# ax.legend(["S11"])
mplm.line_marker(x=f0 / 1e6)
ax.grid()

plt.show()