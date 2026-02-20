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

ms_w = 0.02
ms_len = 0.5
ms1_y = 0.2
ms2_y = -0.2
slot_gap = 0.005

sbox_h = 0.4
sbox_w = 1
sbox_len = ms_len * 2.5

sub_h = 0.01
ms_x = (-ms_len, 0)

f0 = 10e9
lam0 = rfn.const.c0_in / f0
er_eff = 2.8
lam_qw = lam0 / (4 * np.sqrt(er_eff))

line_ref = rfn.elements.MSLine(h=sub_h, er=3.66, w=ms_w, length=ms_len * 1.0)
z_ref = line_ref.get_properties(10e9).sel(value="z0").item()


substrate = pv.Box(bounds=(-sbox_len/2, sbox_len/4, -sbox_w/2, sbox_w/2, 0, sub_h))

sbox = pv.Box(bounds=(-sbox_len/2, sbox_len/4, -sbox_w/2, sbox_w/2, -sbox_h/2, sbox_h/2))

ms1_trace = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[1] + slot_gap /2, ms1_y + ms_w/2, sub_h)
])

port1_face = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, 0),
])

ms2_trace = pv.Rectangle([
    (ms_x[0], ms2_y - ms_w/2, sub_h),
    (ms_x[0], ms2_y + ms_w/2, sub_h),
    (ms_x[1] + slot_gap /2, ms2_y + ms_w/2, sub_h)
])

port2_face = pv.Rectangle([
    (ms_x[0], ms2_y - ms_w/2, sub_h),
    (ms_x[0], ms2_y + ms_w/2, sub_h),
    (ms_x[0], ms2_y + ms_w/2, 0),
])


gnd_plane_left = pv.Rectangle([
    (-sbox_len/2, -sbox_w/2, 0),
    (-sbox_len/2, sbox_w/2, 0),
    (0 - slot_gap/2, sbox_w/2, 0),
])

gnd_plane_right = pv.Rectangle([
    (sbox_len/4, -sbox_w/2, 0),
    (sbox_len/4, sbox_w/2, 0),
    (slot_gap/2, sbox_w/2, 0),
])

slot_back1 = pv.Rectangle([
    (-slot_gap/2, sbox_w/2, 0),
    (-slot_gap/2, ms1_y + lam_qw, 0),
    (slot_gap/2, ms1_y + lam_qw, 0),
])

slot_back2 = pv.Rectangle([
    (-slot_gap/2, -sbox_w/2, 0),
    (-slot_gap/2, ms2_y - lam_qw, 0),
    (slot_gap/2, ms2_y - lam_qw, 0),
])


current_face = pv.Rectangle([
    (-slot_gap*10, 0, 0.005),
    (-slot_gap*10, 0, -0.005),
    (slot_gap*10, 0, -0.005),
])


voltage_line = pv.Line(
    [-slot_gap/2, 0, 0], [slot_gap/2, 0, sub_h]
)


via1 = pv.Box(bounds=(ms_x[1] + slot_gap/2, ms_x[1] + 0.01 + slot_gap/2, ms1_y-0.01, ms1_y+0.01, 0, sub_h))
via2 = pv.Box(bounds=(ms_x[1] + slot_gap/2, ms_x[1] + 0.01 + slot_gap/2, ms2_y-0.01, ms2_y+0.01, 0, sub_h))

conductors = [ms1_trace, ms2_trace, via1, via2, ]

s = rfn.FDTD_Solver(sbox)
s.add_dielectric(substrate, er=3.66, style=dict(opacity=0.1))
s.add_conductor((ms1_trace, ms2_trace, via1, via2), style=dict(color="gold"))
s.add_conductor(
    (gnd_plane_left, gnd_plane_right, slot_back1, slot_back2), style=dict(color="grey", opacity=1)
)
s.add_lumped_port(1, port1_face)
s.add_lumped_port(2, port2_face)

s.assign_PML_boundaries("z+", "z-", "y-", "y+", n_pml=5)

self = s

s.generate_mesh(d0 = 0.02, d_edge=0.0025)
# s.edge_correction(ms1_trace)
# s.edge_correction(ms2_trace)

s.add_field_monitor("mon1", "ex", "z", 0, 30)
s.add_field_monitor("mon2", "ez", "z", 0, 30)

s.add_current_probe("c1", current_face)
s.add_line_probe("v1", "ex", voltage_line)


plotter = s.render(show_probes=True)
plotter.camera_position = "xy"
plotter.show()

# Db_0 = s.dt / u0
# Cb_0 = s.dt / e0 
# p = s.plot_coefficients("ex_y", "a", "z", 0, point_size=15, cmap="brg")
# p.camera_position = "xy"
# p.show()

pulse_n = 2800
# width of half pulse in time
pulse_width = (s.dt * 400)
# center of the pulse in time
t0 = (s.dt * 500)

vsrc = 1e-2 * self.gaussian_modulated_source(f0, width=400e-12, t0=200e-12, t_len=600e-12)
plt.plot(vsrc)


s.run([1], [vsrc], n_threads=4)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

# compute slot impedance
slot_i = self.vi_probe_values("c1")
slot_v = self.vi_probe_values("v1")

IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(-self.vi_probe_values("v1"), frequency, 1 / s.dt)
ZP = VP / IP

sdata = s.get_sparameters(frequency, downsample=False)
S11 = sdata[:, 0]
S21 = sdata[:, 1]

p = s.plot_monitor(["mon2"], el=50, zoom=2, az=-10, view="xz", vmin=10, vmax=35, opacity=["linear", "linear"], linear=False, cmap="jet", style="surface")
p.show(title="EM Solver")

# fig, ax = plt.subplots()
# plt.plot(frequency / 1e9, ZP.real)
# ax.set_xlabel("Frequency [GHz]")
# ax.set_ylabel("Impedance [Ohm]")


fig, ax = plt.subplots()

ax.plot(frequency / 1e9, conv.db20_lin(S11))
ax.plot(frequency / 1e9, conv.db20_lin(S21))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S11", "S21"])




fig.tight_layout()
plt.show()
