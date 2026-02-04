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

ms_w = 0.040
ms_x = 0

sbox_h = 0.3
sbox_w = 0.4
sbox_len = 1

sub_h = 0.02
ms_y = (-sbox_len/2, sbox_len/2 - 0.1)

line = rfn.elements.MSLine(h=sub_h, er=3.66, w=ms_w)
z_ref = line.get_properties(10e9).sel(value="z0").item()


substrate = pv.Cube(center=(0, 0, sub_h/2), x_length=sbox_w, y_length=sbox_len, z_length=sub_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)

ms1_trace = pv.Rectangle([
    (ms_x - ms_w/2, ms_y[0], sub_h),
    (ms_x + ms_w/2, ms_y[0], sub_h),
    (ms_x + ms_w/2, ms_y[1], sub_h)
])

port1_face = pv.Rectangle([
    (ms_x - ms_w/2, ms_y[1], sub_h),
    (ms_x + ms_w/2, ms_y[1], sub_h),
    (ms_x + ms_w/2, ms_y[1], 0),
])

s = rfn.Solver_3D(sbox)
s.add_dielectric("sub", substrate, er=3.66, style=dict(opacity=0.0))
s.add_conductor("ms1", ms1_trace, style=dict(color="gold"))
s.add_lumped_port(1, port1_face)
self = s
s.assign_PML_boundaries("y-")


s.generate_mesh(d0 = 0.02, d_edge=0.005)




s.add_field_monitor("mon1", "ez", "y", 0, 15)
s.add_field_monitor("mon2", "ey", "z", sub_h, 10)
s.add_field_monitor("mon3", "ex", "z", sub_h, 10)



plotter = s.render(show_probes=True)
plotter.camera_position = "yz"
plotter.show()


# Db_0 = s.dt / u0
# Cb_0 = s.dt / e0 
p = s.plot_coefficients("ez_y", "a", "y", ms_y[1], point_size=15, cmap="brg")
p.camera_position = "xy"
p.show()

f0 = 10e9
pulse_n = 1400
# width of half pulse in time
t_half = (s.dt * 150)
# center of the pulse in time
t0 = (s.dt * 400)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
plt.plot(vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]



p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface")
p.show(title="EM Solver")

# compute line impedance
# line_i = self.vi_probe_values("c1")
# line_v = self.vi_probe_values("v1")

# IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
# VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
# ZP = VP / IP

fig, ax = plt.subplots()
rfn.plots.draw_smithchart(ax)
plt.plot(S11.real, S11.imag)

fig, ax = plt.subplots()
plt.plot(frequency / 1e9, conv.db20_lin(S11))
plt.plot(frequency / 1e9, conv.db20_lin(S21))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S11", "S21"])

plt.show()