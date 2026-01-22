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

ms_w = 0.01
ms1_y = 0

sbox_h = 0.25
sbox_w = 0.4
sbox_len = 1

sub_h = 0.02
ms_x = (-sbox_len/2 + 0.1, sbox_len/2)

line = rfn.elements.MSLine(h=sub_h, er=3.66, w=ms_w)
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


current_face = pv.Rectangle([
    (-0.25, ms1_y - ms_w/2 - 0.001, sub_h + 0.001),
    (-0.25, ms1_y + ms_w/2 + 0.001, sub_h + 0.001),
    (-0.25, ms1_y + ms_w/2 + 0.001, sub_h - 0.001),
])


voltage_line1 = pv.Line(
    [-0.25, ms1_y, 0], [-0.25, ms1_y, sub_h]
)

voltage_line2 = pv.Line(
    [0.25, ms1_y, 0], [0.25, ms1_y, sub_h]
)

s = rfn.Solver_PCB(sbox, nports=1)
s.add_substrate("sub", substrate, er=3.66, opacity=0.0)
s.add_pec_face("ms1", ms1_trace, color="gold")
s.add_lumped_port(1, port1_face)

self = s

d0 = 0.02
d_pec = 0.01
n_min_pec=4
d_sub=0.01
n_min_sub=4
n0 = 2

s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=3, d_sub=0.005, n_min_sub=2, blend_pec=False)
s.init_coefficients()

s.init_ports()
s.add_xPML(side="upper")
s.init_pec(edge_correction=True)

s.add_field_monitor("mon1", "hz", "z", sub_h, 10)
s.add_field_monitor("mon2", "ey", "z", sub_h, 10)
s.add_field_monitor("mon3", "ex", "z", sub_h, 10)

s.add_current_probe("c1", current_face)
s.add_voltage_probe("v1", voltage_line1)
s.add_voltage_probe("v2", voltage_line2)


# plotter = s.render(show_probes=True)
# plotter.camera_position = "yz"
# plotter.show()


Db_0 = s.dt / u0
Cb_0 = s.dt / e0 
p = s.plot_cooeficients("hz_y2", "b", "z", sub_h, point_size=15, cmap="brg", normalization=Db_0)
p.camera_position = "xy"
p.show()

f0 = 10e9
pulse_n = 1600
# width of half pulse in time
t_half = (s.dt * 250)
# center of the pulse in time
t0 = (s.dt * 500)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]


p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="points")
p.show(title="EM Solver")

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v1 = self.vi_probe_values("v1")
line_v2 = self.vi_probe_values("v2")

def get_vp(v1, v2, d):
    """
    propagation velocity determined by the voltage waves observed at two points separated by distance d, in meters
    """
    corr = np.convolve(v1, np.flip(v2), mode="same")
    delta_n = ((len(v1) / 2)) - np.argmax(np.abs(corr))
    # plt.plot(v1)
    # plt.plot(v2)
    # print(np.argmax(v2) - np.argmax(v1))
    delta_t = s.dt * delta_n

    return np.abs(d) / delta_t

vp_e = get_vp(line_v1, line_v2, conv.m_in(0.5))

print("vp / c", vp_e / rfn.const.c0)


IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
ZP = VP / IP

fig, ax = plt.subplots()
plt.plot(frequency / 1e9, ZP.real)
# ax.plot(frequency / 1e9, conv.z_gamma(S11))
plt.ylim([0, 120])
plt.axhline(y=z_ref, linestyle=":", color="k")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
mplm.line_marker(x = 10, axes=ax)

S11_z = conv.gamma_z(ZP)

fig, ax = plt.subplots()
rfn.plots.draw_smithchart(ax)
plt.plot(S11.real, S11.imag)
plt.plot(S11_z.real, S11_z.imag)

# fig, ax = plt.subplots()
# plt.plot(frequency, conv.db20_lin(S11))
plt.show()