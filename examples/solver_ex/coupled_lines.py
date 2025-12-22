import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import rfnetwork as rfn
import mpl_markers as mplm


import sys
# matplotlib.use("qt5agg")

pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)

sys.argv = sys.argv[0:1]

u0 = const.u0
e0 = const.e0
c0 = const.c0


# remove e edge conductive on x sides as well if there is no port attached to it
# allocate block memory in python and assign section to each thread, that way multiple runs can share the same
# memory without reallocating it.

ms_w = 0.03
ms_sp = 0.005

ms1_y = -(ms_w / 2) - (ms_sp / 2)
ms2_y = (ms_w / 2) + (ms_sp / 2)

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

ms2_trace = pv.Rectangle([
    (ms_x[0], ms2_y - ms_w/2, sub_h),
    (ms_x[0], ms2_y + ms_w/2, sub_h),
    (ms_x[1], ms2_y + ms_w/2, sub_h)
])

port1_face = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, 0),
])

port2_face = pv.Rectangle([
    (ms_x[0], ms2_y - ms_w/2, sub_h),
    (ms_x[0], ms2_y + ms_w/2, sub_h),
    (ms_x[0], ms2_y + ms_w/2, 0),
])


current_face = pv.Rectangle([
    (0, ms1_y - ms_w/2, sub_h + 0.001),
    (0, ms1_y + ms_w/2, sub_h + 0.001),
    (0, ms1_y + ms_w/2, sub_h - 0.001),
])


voltage_line = pv.Line(
    [0, ms1_y, 0], [0, ms1_y, sub_h]
)

s = rfn.Solver_PCB(sbox, nports=2)
s.add_substrate("sub", substrate, er=3.66, opacity=0.0)
s.add_pec_face("ms1", ms1_trace, color="gold")
s.add_pec_face("ms2", ms2_trace, color="gold")
s.add_lumped_port(1, port1_face)
s.add_lumped_port(2, port2_face)
# s.add_lumped_port(2, port2_face)

self = s

d0 = 0.02
d_pec = 0.01
n_min_pec=4
d_sub=0.01
n_min_sub=4

s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=3, d_sub=0.005, n_min_sub=4, blend_pec=False)
s.init_coefficients()

s.init_ports()
s.add_xPML(side="upper")
s.init_pec()

s.add_field_monitor("mon1", "ez", "z", sub_h - 0.005, 10)
s.add_field_monitor("mon2", "ey", "z", sub_h, 10)
s.add_field_monitor("mon3", "ex", "z", sub_h, 10)

s.add_current_probe("c1", current_face)
s.add_voltage_probe("v1", voltage_line)


plotter = s.render(show_probes=True)
plotter.camera_position = "yz"
plotter.show()


# # print(s.Nx, s.Ny, s.Nz)

# # p.camera.zoom(1.8)

# Db_0 = s.dt / u0
# Cb_0 = s.dt / e0 
# # p = s.plot_cooeficients("ey_x", "b", "z", sub_h, point_size=15, cmap="brg", normalization=Cb_0)
# # p.camera_position = "xy"
# # p.show()

f0 = 10e9
pulse_n = 1200
# width of half pulse in time
t_half = (s.dt * 250)
# center of the pulse in time
t0 = (s.dt * 500)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)


frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

# run even mode
s.run([1, 2], [vsrc, vsrc], n_threads=4)

sdata = s.get_sparameters(frequency)
S11_even = sdata[:, 0]

# run odd mode
s.run([1, 2], [vsrc, -vsrc], n_threads=4)


sdata = s.get_sparameters(frequency)
S11_odd = sdata[:, 0]

# p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=True, cmap="RdBu", style="surface")
# p.show(title="EM Solver")


# fig, ax = plt.subplots()
# rfn.plots.draw_smithchart(ax)
# plt.plot(S11.real, S11.imag)

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v = self.vi_probe_values("v1")

IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
ZP = VP / IP

fig, ax = plt.subplots()
# plt.plot(frequency / 1e9, ZP.real)
ax.plot(frequency / 1e9, conv.z_gamma(S11_even))
ax.plot(frequency / 1e9, conv.z_gamma(S11_odd))
plt.ylim([0, 110])
# plt.axhline(y=z_ref, linestyle=":", color="k")
plt.axhline(y=72, linestyle=":", color="k")
plt.axhline(y=45, linestyle=":", color="k")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
ax.legend(["Even Mode", "Odd Mode"])
mplm.line_marker(x = 10, axes=ax)

# S11_z = conv.gamma_z(ZP)

# fig, ax = plt.subplots()
# rfn.plots.draw_smithchart(ax)
# plt.plot(S11.real, S11.imag)
# plt.plot(S11_z.real, S11_z.imag)

# fig, ax = plt.subplots()
# plt.plot(frequency, conv.db20_lin(S11))
plt.show()