import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time

import rfnetwork as rfn
import mpl_markers as mplm
import sys
from pathlib import Path

dir_ = Path(__file__).parent

DATA_DIR = dir_ / "data"


pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)

sys.argv = sys.argv[0:1]

u0 = const.u0
e0 = const.e0
c0 = const.c0

# 0.03
ms_w = 0.03
ms1_y = 0
er = 1.001

sbox_h = 0.25
sbox_w = 0.4
sbox_len = 1

sub_h = 0.02
ms_x = (-sbox_len/2 + 0.1, sbox_len/2)


substrate = pv.Cube(center=(0, 0, sub_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sub_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

ms1_trace = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[1], ms1_y + ms_w/2, sub_h)
])

ms = pv.Cube(bounds=(ms_x[0], ms_x[1], ms1_y - ms_w/2, ms1_y + ms_w/2, sub_h, sub_h + 0.005))

port1_face = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, 0),
])

s = rfn.Solver_3D(sbox)


s.add_dielectric("sub", substrate, er=er, style=dict(opacity=0.1))

s.add_conductor("ms1", ms, style=dict(color="gold", opacity=0.5))
s.add_lumped_port(1, port1_face)

s.assign_PML_boundaries("x+", "z+")

self = s

s.generate_mesh(d0=0.01, d_edge=0.005)

plotter = s.render(show_probes=False)
plotter.camera_position = "yz"
plotter.show()
print(s.Nx * s.Ny * s.Nz / 1e3)


p = s.plot_coefficients("ez_x", "a", "x", 0, point_size=15, cmap="brg")
p.camera_position = "yz"
p.show()


s.add_field_monitor("ez", "ez", "z", sub_h, 1)
s.add_field_monitor("ey", "ey", "z", sub_h, 1)
s.add_field_monitor("ex", "ex", "z", sub_h, 1)


f0 = 10e9
pulse_n = 1600
# width of half pulse in time
t_half = (s.dt * 250)
# center of the pulse in time
t0 = (s.dt * 500)

# t = np.linspace(0, s.dt * pulse_n, pulse_n)
# vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

vsrc = 1e-2 * self.gaussian_modulated_source(f0, width=t_half*3, t0=t0, t_len=pulse_n*s.dt)
plt.plot(vsrc)


frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]


p = s.plot_monitor(["ey"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface", vmax=20)
p.show(title="EM Solver")
