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

ms_w = 0.04
ms_len = 1
ms1_y = 0

sbox_h = 0.3
sbox_w = 0.4
sbox_len = ms_len * 1.3

sub_h = 0.02
ms_x = ((-ms_len/2), (ms_len/2))

line_ref = rfn.elements.MSLine(h=sub_h, er=3.66, w=ms_w, length=ms_len * 1.0)
z_ref = line_ref.get_properties(10e9).sel(value="z0").item()


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

port2_face = pv.Rectangle([
    (ms_x[1], ms1_y - ms_w/2, sub_h),
    (ms_x[1], ms1_y + ms_w/2, sub_h),
    (ms_x[1], ms1_y + ms_w/2, 0),
])


current_face = pv.Rectangle([
    (0, ms1_y - ms_w/2 - 0.001, sub_h + 0.001),
    (0, ms1_y + ms_w/2 + 0.001, sub_h + 0.001),
    (0, ms1_y + ms_w/2 + 0.001, sub_h - 0.001),
])


voltage_line = pv.Line(
    [0, ms1_y, 0], [0, ms1_y, sub_h]
)

s = rfn.Solver_PCB(sbox, nports=2)
s.add_substrate("sub", substrate, er=3.66, opacity=0.0)
s.add_pec_face("ms1", ms1_trace, color="gold")
s.add_lumped_port(1, port1_face)
s.add_lumped_port(2, port2_face)

self = s

d0 = 0.02
d_pec = 0.01
n_min_pec=4
d_sub=0.01
n_min_sub=4
n0 = 2

# having three cells in the PEC instead of 4 causes the edge correction to fail
# s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=4, d_sub=0.01, n_min_sub=4, blend_pec=False)
s.init_mesh_edge_method(d0 = 0.05, d_edge=0.005)
s.init_coefficients()

s.init_ports()
s.init_pec(edge_correction=False)
# s.add_xPML(side="upper")

s.add_field_monitor("mon1", "ez", "y", 0, 5)
# s.add_field_monitor("mon1", "ez", "z", sub_h - 0.005, 15)
s.add_field_monitor("mon2", "ey", "z", sub_h, 15)
s.add_field_monitor("mon3", "ex", "z", sub_h, 10)

s.add_current_probe("c1", current_face)
s.add_voltage_probe("v1", voltage_line)


# plotter = s.render(show_probes=True)
# plotter.camera_position = "yz"
# plotter.show()


Db_0 = s.dt / u0
Cb_0 = s.dt / e0 
# p = s.plot_cooeficients("hy_x", "b", "z", sub_h - 0.005, point_size=15, cmap="brg", normalization=Db_0)
# p.camera_position = "xy"
# p.show()

f0 = 10e9
pulse_n = 2800
# width of half pulse in time
t_half = (s.dt * 150)
# center of the pulse in time
t0 = (s.dt * 400)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]
S21 = sdata[:, 1]


p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface")
p.show(title="EM Solver")

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v = self.vi_probe_values("v1")

IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
ZP = VP / IP
S11_zp = rfn.conv.gamma_z(ZP)

sdata_ref = line_ref.evaluate(frequency)["s"] 

fig, ax = plt.subplots()
plt.plot(frequency / 1e9, ZP.real)
ax.plot(frequency / 1e9, conv.z_gamma(S11))
plt.ylim([0, 120])
plt.axhline(y=z_ref, linestyle=":", color="k")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
mplm.line_marker(x = 10, axes=ax)
ax.legend(["probe", "port"])


fig, axes = plt.subplots(2, 2, figsize=(9, 9))

ax = axes[0,0]
rfn.plots.draw_smithchart(ax)
ax.plot(S11.real, S11.imag)
ax.plot(sdata_ref.sel(b=1, a=1).real, sdata_ref.sel(b=1, a=1).imag)



ax = axes[0,1]
ax.plot(frequency / 1e9, conv.db20_lin(S11))
ax.plot(frequency / 1e9, conv.db20_lin(sdata_ref).sel(b=1, a=1))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S11", "Ref"])

ax = axes[1,0]
ax.plot(frequency / 1e9, conv.db20_lin(S21))
ax.plot(frequency / 1e9, conv.db20_lin(sdata_ref).sel(b=2, a=1))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S21", "Ref"])

ax = axes[1,1]
ax.plot(frequency / 1e9, np.unwrap(np.angle(S21, deg=True)))
ax.plot(frequency / 1e9, np.unwrap(np.angle(sdata_ref.sel(b=2, a=1), deg=True)))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[deg]")
ax.legend(["S21", "Ref"])


fig.tight_layout()
plt.show()