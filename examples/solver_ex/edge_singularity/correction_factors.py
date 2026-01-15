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

sbox_h = 0.2
sbox_w = 0.3
sbox_len = 0.25

sub_h = 0.02
ms_x = (-sbox_len/2 + 0.1, sbox_len/2)

line = rfn.elements.MSLine(h=sub_h, er=er, w=ms_w)
z_ref = line.get_properties(10e9).sel(value="z0").item()

substrate = pv.Cube(center=(0, 0, sub_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sub_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

ms1_trace = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[1], ms1_y + ms_w/2, sub_h)
])

port1_face = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, 0),
])

hy_ms_edge = pv.Line(
    [0, ms_w/2, 0], [0, ms_w/2, sub_h * 2]
)

hy_ms_center = pv.Line(
    [0, 0, 0], [0, 0, sub_h * 2]
)

s = rfn.Solver_PCB(sbox, nports=1)
s.add_substrate("sub", substrate, er=er, opacity=0.0)
s.add_pec_face("ms1", ms1_trace, color="gold")
s.add_lumped_port(1, port1_face)

self = s

d0 = 0.005
d_pec = 0.001
s.init_mesh(d0 = d0, n0 = 3, d_pec = d_pec, n_min_pec=3, d_sub=d_pec, n_min_sub=2, blend_pec=False)
s.init_coefficients()

s.init_ports()
s.add_xPML(side="upper")

s.init_pec()

s.add_field_monitor("ez", "ez", "z", sub_h, 1)

s.add_line_probe("hy_ms_edge", "hy", hy_ms_edge)
s.add_line_probe("hy_ms_center", "hy", hy_ms_center)



plotter = s.render(show_probes=True)
plotter.camera_position = "yz"
plotter.show()

s.Nx * s.Ny * s.Nz / 1e6



vsrc = 1e-2 * self.gaussian_source(width=15e-12, t_len=30e-12)
t = np.linspace(0, self.dt * len(vsrc), len(vsrc))
plt.plot(t / 1e-12, vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]


p = s.plot_monitor(["ez"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="points")
p.show(title="EM Solver")

t_sample = 18e-12
n_sample = int(t_sample / self.dt)

hy_ms_edge = self.line_probe_values("hy_ms_edge")
hy_ms_center = self.line_probe_values("hy_ms_center")
hy_loc = s.floc["hy"][2][:len(hy_ms_edge)]

plt.figure()
plt.plot(hy_loc,  hy_ms_edge[:, n_sample], marker=".")
plt.plot(hy_loc,  hy_ms_center[:, n_sample], marker=".")
plt.show()
