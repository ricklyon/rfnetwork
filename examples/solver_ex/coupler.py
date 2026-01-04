import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import rfnetwork as rfn
import mpl_markers as mplm
from scipy.optimize import least_squares


import sys
# matplotlib.use("qt5agg")

pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)

sys.argv = sys.argv[0:1]

u0 = const.u0
e0 = const.e0
c0 = const.c0

# design parameters
f0 = 3e9
er = 2.2
Z0 = 50
b = 0.125

# wavelength at design frequency
lam0_in = rfn.const.c0_in / (f0 * np.sqrt(er))

# coupling factor, db
C_db = 20
C = 10 **(-np.abs(C_db) / 20)

# even, odd mode impedance
Z0_e = Z0 * np.sqrt((1 + C) / (1 - C))
Z0_o = Z0 * np.sqrt((1 - C) / (1 + C))

# even ad odd mode capacitances, normalized by epsilon
# Z0_e = 1 / vp*Ce, Table 5.05-1 (2) if Ca=Cb
# Z0_o = 1 / vp*Co
# Co = Ca + 2 Cab
# Ce = Ca = Cb
vp = const.c0 / np.sqrt(er)
Ce = 1 / (Z0_e * vp * const.e0 * er)
Co = 1 / (Z0_o * vp * const.e0 * er)

# Co = Ca + 2 Cab
# Ca = Ce
# Cab = (Co - Ca) / 2
Cab = (Co - Ce) / 2

def find_spacing(sp, target_Cab):
   Cf_o, Cf_e = utils.coupled_sline_fringing_cap(b * 0.5, sp, b, er)
   # normalized Cab
   Cab = (Cf_o - Cf_e) / (e0 * er)

   return Cab - target_Cab

# determine spacings using the capacitance between lines Cmk
sp = least_squares(find_spacing, x0=b*0.2, args=(Cab,), bounds=(0.001, b)).x[0]

_, Cf_e = utils.coupled_sline_fringing_cap(b * 0.5, sp, b, er)

Cf_e = Cf_e / (e0 * er)
# fringing capacitance on the outer edges (not between the two lines), figure 5.05-10b, for t=0
Cf = 0.44
# parallel plate capacitance
Cp_e = (Ce / 2) - Cf - Cf_e
# determine width using parallel plate capacitance Cp_e = 2w / b
w = (Cp_e / 2) * b


w_ref = 0.809 * (b)
s_ref = 0.306 * (b)


s1_y = -(w / 2) - (sp / 2)
s2_y = (w / 2) + (sp / 2)

sbox_h = b
sbox_w = 0.8
sbox_len = (lam0_in / 2)

sub_h = (b / 2)
s_x = (-(lam0_in / 8), (lam0_in / 8))

substrate = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)


s1_trace = pv.Rectangle([
    (s_x[0], s1_y - w/2, sub_h),
    (s_x[0], s1_y + w/2, sub_h),
    (s_x[1], s1_y + w/2, sub_h)
])

s2_trace = pv.Rectangle([
    (s_x[0], s2_y - w/2, sub_h),
    (s_x[0], s2_y + w/2, sub_h),
    (s_x[1], s2_y + w/2, sub_h)
])

# ports on -x side of traces
port1a_face = pv.Rectangle([
    (s_x[0], s1_y - w/2, sub_h),
    (s_x[0], s1_y + w/2, sub_h),
    (s_x[0], s1_y + w/2, 0),
])

port1b_face = pv.Rectangle([
    (s_x[0], s1_y - w/2, sub_h),
    (s_x[0], s1_y + w/2, sub_h),
    (s_x[0], s1_y + w/2, b),
])

port3a_face = pv.Rectangle([
    (s_x[0], s2_y - w/2, sub_h),
    (s_x[0], s2_y + w/2, sub_h),
    (s_x[0], s2_y + w/2, 0),
])

port3b_face = pv.Rectangle([
    (s_x[0], s2_y - w/2, sub_h),
    (s_x[0], s2_y + w/2, sub_h),
    (s_x[0], s2_y + w/2, b),
])

# ports on +x side of traces
port2a_face = pv.Rectangle([
    (s_x[1], s1_y - w/2, sub_h),
    (s_x[1], s1_y + w/2, sub_h),
    (s_x[1], s1_y + w/2, 0),
])

port2b_face = pv.Rectangle([
    (s_x[1], s1_y - w/2, sub_h),
    (s_x[1], s1_y + w/2, sub_h),
    (s_x[1], s1_y + w/2, b),
])

port4a_face = pv.Rectangle([
    (s_x[1], s2_y - w/2, sub_h),
    (s_x[1], s2_y + w/2, sub_h),
    (s_x[1], s2_y + w/2, 0),
])

port4b_face = pv.Rectangle([
    (s_x[1], s2_y - w/2, sub_h),
    (s_x[1], s2_y + w/2, sub_h),
    (s_x[1], s2_y + w/2, b),
])

current_face = pv.Rectangle([
    (-0.25, s1_y - w/2 - 0.001, sub_h + 0.001),
    (-0.25, s1_y + w/2 + 0.001, sub_h + 0.001),
    (-0.25, s1_y + w/2 + 0.001, sub_h - 0.001),
])


voltage_line1 = pv.Line(
    [-0.25, s1_y, 0], [-0.25, s1_y, sub_h]
)

voltage_line2 = pv.Line(
    [0.25, s1_y, 0], [0.25, s1_y, sub_h]
)

s = rfn.Solver_PCB(sbox, nports=8)
s.add_substrate("sub", substrate, er=er, opacity=0.0)
s.add_pec_face("ms1", s1_trace, color="gold")
s.add_pec_face("ms2", s2_trace, color="gold")
s.add_lumped_port(1, port1a_face)
s.add_lumped_port(2, port1b_face)

s.add_lumped_port(3, port2a_face)
s.add_lumped_port(4, port2b_face)

s.add_lumped_port(5, port3a_face)
s.add_lumped_port(6, port3b_face)

s.add_lumped_port(7, port4a_face)
s.add_lumped_port(8, port4b_face)

self = s


s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=3, d_sub=0.01, n_min_sub=4, blend_pec=False)
s.init_coefficients()

s.init_ports(r0=100)
s.add_xPML(side="upper")
s.add_xPML(side="lower")
s.init_pec(edge_correction=False)

s.add_field_monitor("mon1", "ez", "z", sub_h, 10)

s.add_current_probe("c1", current_face)
s.add_voltage_probe("v1", voltage_line1)
s.add_voltage_probe("v2", voltage_line2)

# plotter = s.render(show_probes=True)
# plotter.camera_position = "yz"
# plotter.show()

pulse_n = 3000
# width of half pulse in time
t_half = (s.dt * 110)
# center of the pulse in time
t0 = (s.dt * 350)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

frequency: np.ndarray = np.arange(1e9, 5e9, 10e6)

# run even mode
# apply signal to port 1
s.run([1, 2], [vsrc, -vsrc], n_threads=4)

# s.run([1, 2, 3, 4], [vsrc, -vsrc, vsrc, -vsrc], n_threads=4)

p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface")
p.show(title="EM Solver")


sdata = s.get_sparameters(frequency, source_port=1, z0=100)
S11 = sdata[:,0]
S21 = sdata[:,2]
S31 = sdata[:,4]
S41 = sdata[:,6]

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v1 = self.vi_probe_values("v1")
line_v2 = self.vi_probe_values("v2")

# plt.plot(line_v1)
# plt.plot(line_v2)

IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
ZP_even = VP / IP




fig, ax = plt.subplots()
# plt.plot(frequency / 1e9, ZP.real)
ax.plot(frequency / 1e9, conv.db20_lin(S11))
ax.plot(frequency / 1e9, conv.db20_lin(S21))
ax.plot(frequency / 1e9, conv.db20_lin(S31))
ax.plot(frequency / 1e9, conv.db20_lin(S41))
ax.legend(["S11", "S21", "S31", "S41"])
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.grid(True)


# S11_z = conv.gamma_z(ZP)

# fig, ax = plt.subplots()
# rfn.plots.draw_smithchart(ax)
# plt.plot(S11.real, S11.imag)
# plt.plot(S11_z.real, S11_z.imag)

# fig, ax = plt.subplots()
# plt.plot(frequency, conv.db20_lin(S11))
plt.show()