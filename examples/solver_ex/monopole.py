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

sbox_h = lam0
sbox_w = lam0
sbox_len = lam0

gap = 0.01
ms_x = (0, 0)
ms_y = (-ms_w / 2, ms_w / 2)
ms_z = (gap, (lam0 / 4) * 0.95)


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


s = rfn.FDTD_Solver(sbox)
# s.add_dielectric("sub", substrate, er=1, style=dict(opacity=0.0))
s.add_conductor(ms1_trace, style=dict(color="gold"))
s.add_lumped_port(1, port1_face, "z-")

s.assign_PML_boundaries("x-", "x+", "y-", "y+", "z+", n_pml=5)

self = s
# having three cells in the PEC instead of 4 causes the edge correction to fail
# s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=4, d_sub=0.01, n_min_sub=4, blend_pec=False)
s.generate_mesh(d0 = 0.03, d_edge=0.005)


s.add_field_monitor("mon1", "e_total", "z", 0.15, 10)


# plotter = s.render(show_probes=True)
# plotter.camera_position = "yz"
# plotter.show()


# Db_0 = s.dt / u0
# Cb_0 = s.dt / e0 
# p = s.plot_coefficients("ey_x", "b", "x", 0, point_size=15, cmap="brg")
# p.show()

f0 = 10e9
pulse_n = 2800
# width of half pulse in time
pulse_width = (s.dt * 400)
# center of the pulse in time
t0 = (s.dt * 500)

vsrc = 1e-2 * self.gaussian_modulated_source(f0, width=600e-12, t0=220e-12, t_len=800e-12)
# plt.plot(vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.assign_excitation(vsrc, 1)
s.solve(n_threads=4)

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]


gif_setup = dict(file="vectors.gif", fps=20)

monitor = "mon1"
view="yz"
style="vectors"
linear=False
name ="mon1"
vmax = None
vmin = None
init_time = None

init_time = None
opacity = "linear"

# fields = s.get_monitor_data("mon1_z")

# fields.sel(time=100e-12, y=0, z=0.15)


plotter = self.plot_monitor(["mon1", "mon1"], camera_position="xy", zoom=1, style=["vectors", "surface"], opacity=0.8, max_vector_len=0.03, show_rulers=False, gif_setup=None)
plotter.show()

# p.show(title="EM Solver")


fig, ax = plt.subplots()
plt.plot(frequency / 1e9, conv.db20_lin(S11))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S11"])

plt.show()