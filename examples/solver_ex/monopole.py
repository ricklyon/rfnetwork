import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time

import rfnetwork as rfn
import mpl_markers as mplm
import sys


pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)

sys.argv = sys.argv[0:1]

u0 = const.u0
e0 = const.e0
c0 = const.c0

f0 = 10e9
lam0 = const.c0_in / f0

ms_w = 0.04
ms_len = 1
ms1_y = 0

sbox_h = lam0 / 2
sbox_w = lam0 / 2
sbox_len = lam0 / 2

sub_h = 0.02
gap = 0.01
ms_x = (0, 0)
ms_y = (-ms_w / 2, ms_w / 2)
ms_z = (gap, gap + (lam0 / 4))

# substrate = pv.Cube(center=(0, 0, sub_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sub_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

ms1_trace = pv.Rectangle([
    (ms_x[0], ms_y[0], ms_z[0]),
    (ms_x[0], ms_y[1], ms_z[0]),
    (ms_x[1], ms_y[1], ms_z[1])
])

port1_face = pv.Rectangle([
    (ms_x[0], ms_y[0], 0),
    (ms_x[0], ms_y[1], 0),
    (ms_x[1], ms_y[1], ms_z[0])
])


s = rfn.Solver_PCB(sbox)
# s.add_dielectric("sub", substrate, er=3.66, style=dict(opacity=0.0))
s.add_conductor("ms1", ms1_trace, style=dict(color="gold"))
s.add_lumped_port(1, port1_face)

s.assign_PML_boundaries("x-", "x+", "y-", "y+", "z+")

self = s
# having three cells in the PEC instead of 4 causes the edge correction to fail
# s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=4, d_sub=0.01, n_min_sub=4, blend_pec=False)
s.generate_mesh(d0 = 0.02, d_edge=0.0025, z_bounds = [0.0025, 0.02])

s.add_field_monitor("mon1", "ey", "x", 0, 5)
s.add_field_monitor("mon2", "ey", "y", 0, 5)

s.add_field_monitor("mon3", "ez", "z", 0.2, 5)



# plotter = s.render(show_probes=True)
# plotter.camera_position = "yz"
# plotter.show()


Db_0 = s.dt / u0
Cb_0 = s.dt / e0 
p = s.plot_coefficients("ez_x", "b", "x", 0, point_size=15, cmap="brg")
p.show()

f0 = 10e9
pulse_n = 2800
# width of half pulse in time
pulse_width = (s.dt * 400)
# center of the pulse in time
t0 = (s.dt * 500)

vsrc = 1e-2 * self.gaussian_modulated_source(f0, width=80e-12, t0=80e-12, t_len=400e-12)
plt.plot(vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]



p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface")
p.show(title="EM Solver")

p = s.plot_monitor(["mon3"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface")
p.show(title="EM Solver")
