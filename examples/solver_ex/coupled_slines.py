import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import rfnetwork as rfn
import mpl_markers as mplm
from scipy.special import ellipk
from scipy import signal

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

s_w = 0.127
s_sp = 0.146

s1_y = -(s_w / 2) - (s_sp / 2)
s2_y = (s_w / 2) + (s_sp / 2)

b = 0.625
sub_h = b / 2
er = 1

sbox_h = b
sbox_w = 1.5
sbox_len = 4


ms_x = (-sbox_len/2 + 0.3, sbox_len/2)

line_ref = rfn.elements.Stripline(w=s_w, b=b, er=er, length=sbox_len - 0.3)
z_ref = line_ref.get_properties(10e9).sel(value="z0").item()

# page 174 in Matthaei, even and odd mode impedances of coupled strip line
k_e = np.tanh((np.pi / 2) * (s_w / b)) * np.tanh((np.pi / 2) * (s_w + s_sp) / b)
kp_e = np.sqrt(1 - (k_e **2))

k_o = np.tanh((np.pi / 2) * (s_w / b)) * (1 / np.tanh((np.pi / 2) * (s_w + s_sp) / b))
kp_o = np.sqrt(1 - (k_o **2))

Z0_e = ((30 * np.pi) / (np.sqrt(er))) * (ellipk(kp_e) / ellipk(k_e))
Z0_o = ((30 * np.pi) / (np.sqrt(er))) * (ellipk(kp_o) / ellipk(k_o))


substrate = pv.Cube(center=(0, 0, sub_h), x_length=sbox_len, y_length=sbox_w, z_length=b)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

s1_trace = pv.Rectangle([
    (ms_x[0], s1_y - s_w/2, sub_h),
    (ms_x[0], s1_y + s_w/2, sub_h),
    (ms_x[1], s1_y + s_w/2, sub_h)
])

s2_trace = pv.Rectangle([
    (ms_x[0], s2_y - s_w/2, sub_h),
    (ms_x[0], s2_y + s_w/2, sub_h),
    (ms_x[1], s2_y + s_w/2, sub_h)
])

port1_face = pv.Rectangle([
    (ms_x[0], s1_y - s_w/2, sub_h),
    (ms_x[0], s1_y + s_w/2, sub_h),
    (ms_x[0], s1_y + s_w/2, 0),
])

port2_face = pv.Rectangle([
    (ms_x[0], s1_y - s_w/2, sub_h),
    (ms_x[0], s1_y + s_w/2, sub_h),
    (ms_x[0], s1_y + s_w/2, b),
])

port3_face = pv.Rectangle([
    (ms_x[0], s2_y - s_w/2, sub_h),
    (ms_x[0], s2_y + s_w/2, sub_h),
    (ms_x[0], s2_y + s_w/2, 0),
])

port4_face = pv.Rectangle([
    (ms_x[0], s2_y - s_w/2, sub_h),
    (ms_x[0], s2_y + s_w/2, sub_h),
    (ms_x[0], s2_y + s_w/2, b),
])

current_face = pv.Rectangle([
    (0, s1_y - s_w/2 - 0.001, sub_h + 0.001),
    (0, s1_y + s_w/2 + 0.001, sub_h + 0.001),
    (0, s1_y + s_w/2 + 0.001, sub_h - 0.001),
])


voltage_line = pv.Line(
    [0, s1_y, 0], [0, s1_y, sub_h]
)


s = rfn.Solver_PCB(sbox, nports=4)
s.add_substrate("sub", substrate, er=er, opacity=0.0)
s.add_pec_face("ms1", s1_trace, color="gold")
s.add_pec_face("ms2", s2_trace, color="gold")
s.add_lumped_port(1, port1_face)
s.add_lumped_port(2, port2_face)
s.add_lumped_port(3, port3_face)
s.add_lumped_port(4, port4_face)
# s.add_lumped_port(2, port2_face)

self = s

f0 = 2e9
lam0 = rfn.const.c0_in / (f0 * np.sqrt(er))

s.init_mesh(d0 = lam0/20, n0 = 4, d_pec = lam0/30, n_min_pec=4, d_sub=lam0/20, n_min_sub=5, blend_pec=False)
s.init_coefficients()

s.init_ports(r0=100)
s.add_xPML(side="upper")
s.init_pec(edge_correction=True)

s.add_field_monitor("mon1", "ez", "z", sub_h - 0.005, 10)

s.add_current_probe("c1", current_face)
s.add_voltage_probe("v1", voltage_line)

# plotter = s.render(show_probes=True)
# plotter.camera_position = "yz"
# plotter.show()

pulse_n = 1200
# width of half pulse in time
t_half = (s.dt * 110)
# center of the pulse in time
t0 = (s.dt * 350)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)


frequency: np.ndarray = np.arange(1e9, 5e9, 10e6)

# run even mode
s.run([1, 2, 3, 4], [vsrc, -vsrc, vsrc, -vsrc], n_threads=4)

sdata = s.get_sparameters(frequency, source_port=2, z0=100)
S11_even = sdata[:,1]

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v = self.vi_probe_values("v1")

plt.plot(line_v)
plt.plot(vsrc)

IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
ZP_even = VP / IP


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

vp_e = get_vp(vsrc, line_v, conv.m_in(ms_x[0]))

print("even vp", vp_e / rfn.const.c0)
# even mode capacitance
c_e = 1 / (vp_e * Z0_e)
print("even c/e", c_e / const.e0)

# p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface")
# p.show(title="EM Solver")

# run odd mode
s.run([1, 2, 3, 4], [vsrc, -vsrc, -vsrc, vsrc], n_threads=4)


sdata = s.get_sparameters(frequency, source_port=2, z0=100)
S11_odd = sdata[:, 1]

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v = self.vi_probe_values("v1")

IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
ZP_odd = VP / IP

vp_o = get_vp(vsrc, line_v,  conv.m_in(ms_x[0]))
print("odd vp", vp_o / rfn.const.c0)

# odd mode capacitance
c_o = 1 / (vp_o * Z0_e)
print("odd c/e", c_o / const.e0)

# mutual capacitance
c_m = (c_o - c_e / 2)
print("c_m", c_m / const.e0)

# p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=True, cmap="jet", style="surface")
# p.show(title="EM Solver")




fig, ax = plt.subplots()
# plt.plot(frequency / 1e9, ZP.real)
ax.plot(frequency / 1e9, np.real(conv.z_gamma(S11_even)))
ax.plot(frequency / 1e9, np.real(conv.z_gamma(S11_odd)))
ax.plot(frequency / 1e9, np.real(ZP_even))
ax.plot(frequency / 1e9, np.real(ZP_odd))
plt.ylim([0, 200])

plt.axhline(y=Z0_e, linestyle=":", color="blue")
plt.axhline(y=Z0_o, linestyle=":", color="orange")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
ax.legend(["Even Mode", "Odd Mode"])
mplm.line_marker(x = 0, axes=ax)

# S11_z = conv.gamma_z(ZP)

fig, ax = plt.subplots()
rfn.plots.draw_smithchart(ax)
plt.plot(S11_even.real, S11_even.imag)
plt.plot(S11_odd.real, S11_odd.imag)

# fig, ax = plt.subplots()
# plt.plot(frequency, conv.db20_lin(S11))
plt.show()