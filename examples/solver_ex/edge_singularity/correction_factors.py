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

ez_ms_edge = pv.Line(
    [0, ms_w/2, 0], [0, ms_w/2, sub_h * 2]
)

ez_ms_center = pv.Line(
    [0, 0, 0], [0, 0, sub_h * 2]
)

hz_ms = pv.Line(
    [0, -ms_w, sub_h], [0, ms_w, sub_h]
)

ey_ms = pv.Line(
    [0, -ms_w, sub_h], [0, ms_w, sub_h]
)

hy_cross_ms = pv.Line(
    [0, -ms_w, sub_h], [0, ms_w, sub_h]
)

hz_v_ms = pv.Line(
    [0, -ms_w/2, 0], [0, -ms_w/2, sub_h]
)



s = rfn.Solver_PCB(sbox, nports=1)
s.add_substrate("sub", substrate, er=er, opacity=0.0)
s.add_pec_face("ms1", ms1_trace, color="gold")
s.add_lumped_port(1, port1_face)

self = s

d0 = 0.005

s.init_mesh_edge_method(d0 = d0, d_edge=0.001)
s.init_coefficients()

s.init_ports()
s.add_xPML(side="upper")

s.init_pec()

s.add_field_monitor("ez", "ez", "z", sub_h, 1)

s.add_line_probe("hy_ms_edge", "hy", hy_ms_edge)
s.add_line_probe("hy_ms_center", "hy", hy_ms_center)

s.add_line_probe("ez_ms_edge", "ez", ez_ms_edge)
s.add_line_probe("ez_ms_center", "ez", ez_ms_center)

s.add_line_probe("hz_ms", "hz", hz_ms)
s.add_line_probe("ey_ms", "ey", ey_ms)

s.add_line_probe("hy_cross_ms", "hy", hy_cross_ms)
s.add_line_probe("ez_cross_ms", "ez", hy_cross_ms)

s.add_line_probe("hz_v_ms", "hz", hz_v_ms)



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


# p = s.plot_monitor(["ez"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="points")
# p.show(title="EM Solver")

t_sample = 18e-12
n_sample = int(t_sample / self.dt)

hy_ms_edge = self.line_probe_values("hy_ms_edge")
hy_ms_center = self.line_probe_values("hy_ms_center")
hy_loc = s.floc["hy"][2][:len(hy_ms_edge)]

ez_ms_edge = self.line_probe_values("ez_ms_edge")
ez_ms_center = self.line_probe_values("ez_ms_center")
ez_loc = s.floc["ez"][2][:len(ez_ms_edge)]

hz_ms = self.line_probe_values("hz_ms")
hz_loc1 = self.field_pos_to_idx((0, -ms_w, sub_h), "hz")[1]
hz_loc2 = self.field_pos_to_idx((0, ms_w, sub_h), "hz")[1]
hz_loc = s.floc["hz"][1][hz_loc1: hz_loc2]

ey_ms = self.line_probe_values("ey_ms")
ey_loc1 = self.field_pos_to_idx((0, -ms_w, sub_h), "ey")[1]
ey_loc2 = self.field_pos_to_idx((0, ms_w, sub_h), "ey")[1]
ey_loc = s.floc["ey"][1][ey_loc1: ey_loc2]

hy_cross_ms = self.line_probe_values("hy_cross_ms")
hyc_loc1 = self.field_pos_to_idx((0, -ms_w, sub_h), "hy")[1]
hyc_loc2 = self.field_pos_to_idx((0, ms_w, sub_h), "hy")[1]
hyc_loc = s.floc["hy"][1][hyc_loc1: hyc_loc2+1]

ez_cross_ms = self.line_probe_values("ez_cross_ms")
ezc_loc1 = self.field_pos_to_idx((0, -ms_w, sub_h), "ez")[1]
ezc_loc2 = self.field_pos_to_idx((0, ms_w, sub_h), "ez")[1]
ezc_loc = s.floc["ez"][1][ezc_loc1: ezc_loc2+1]


hz_v_ms = self.line_probe_values("hz_v_ms")
hz_v_loc = s.floc["hz"][2][:len(hz_v_ms)]

def analytical_edge(x, x0=0, y0=1, f0=1, v=1/2):
    r = (f0 / (x - x0) ** v) 
    return r - r[-1] + y0

plt.figure()
plt.plot(hy_loc,  hy_ms_edge[:, n_sample], marker=".")
plt.plot(hy_loc,  analytical_edge(hy_loc, x0=sub_h, f0=-0.0013, y0=hy_ms_edge[:, n_sample][-1]))
# plt.plot(hy_loc,  hy_ms_center[:, n_sample], marker=".")
plt.show()

plt.figure()
plt.plot(ez_loc,  ez_ms_edge[:, n_sample], marker=".")
plt.plot(ez_loc,  analytical_edge(ez_loc, x0=sub_h, f0=0.55, y0=ez_ms_edge[:, n_sample][-1]))

# numerical correction factor
ez_idx = np.argmin(np.abs(ez_loc - sub_h)) + 1
ez_zloc = ez_loc[ez_idx]
ez_w = np.diff(ez_loc)
cell_w = ez_w[ez_idx-1] / 2 + ez_w[ez_idx] / 2
ez_v = ez_ms_edge[:, n_sample][ez_idx]

f0 = ez_v * (cell_w / 2) ** (1/2)
# f_r = (f0 / (ez_loc - sub_h) ** (1/2)) 

r = np.linspace(0, sub_h * 2, 501)
f_r = (f0 / (r) ** (1/2)) 

plt.figure()
plt.plot(ez_loc,  ez_ms_edge[:, n_sample], marker=".")
plt.plot(r + sub_h,  f_r)

# integrate along one cell width
r_cell = np.linspace(1e-8, cell_w, 50001)
f_r_cell = (f0 / (r_cell) ** (1/2)) 

cf = np.trapezoid(f_r_cell, r_cell) / (ez_v * cell_w)

plt.figure()
plt.plot(hz_loc,  hz_ms[:, n_sample], marker=".")
plt.plot(hz_loc, analytical_edge(hz_loc, x0=ms_w/2, f0=0.002, y0=hz_ms[:, n_sample][-1]))
plt.show()

plt.figure()
plt.plot(ey_loc,  ey_ms[:, n_sample], marker=".")
plt.plot(ey_loc, analytical_edge(ey_loc, x0=ms_w/2, f0=1, y0=ey_ms[:, n_sample][-1]))
plt.show()

# hy along y above and below the trace are singular just above and below the edge, but the Ex field that integrates
# these fields is zero at the trace edge. The Ez field uses these as well at the trace edge which causes error.
plt.figure()
plt.plot(hyc_loc,  hy_cross_ms[:, n_sample], marker=".")
plt.show()

# ez along y above and below the trace
ez_v = ez_cross_ms[:, n_sample]
ez_loc_i = self.field_pos_to_idx((0, -ms_w, sub_h), "ez")
dz = 2 * (self.floc["ez"][2][ez_loc_i[2]] - sub_h)
a = ezc_loc - (-ms_w/2)
# cell width along y
dy_all = np.diff(ezc_loc)
dy = dy_all[np.argmin(np.abs(a))-1] / 2 + dy_all[np.argmin(np.abs(a))] / 2

ez0 = ez_v[np.argmin(np.abs(a))] * (dz /2)
f_a = ez0 * ((dz /2) / (a**2 + (dz /2)**2))

# integrate along one cell
a_cell = np.linspace(-dy / 2, dy/2, 1001)
f_a_cell = ez0 * ((dz /2) / (a_cell**2 + (dz /2)**2))

cf = np.trapezoid(f_a_cell, a_cell) / (ez_v[np.argmin(np.abs(a))] * dy)

cf_an = (dz / dy) * np.arctan(dy/dz)
(dy / dz) * np.arctan(dz/dy)

plt.figure()
plt.plot(ezc_loc,  ez_cross_ms[:, n_sample], marker=".")
plt.plot(ezc_loc, f_a)
plt.plot(a_cell -ms_w/2, f_a_cell)
plt.show()

# a = np.linspace(-0.01, 0, 1001)
# f_a = ez0 * (ez_z / (a**2 + ez_z**2))

plt.figure()
plt.plot(ezc_loc,  ez_cross_ms[:, n_sample], marker=".")
plt.plot(a - ms_w/2, f_a)
plt.xlabel("y [in]")
plt.ylabel("Ez [V/m]")
plt.show()


# numerical correction factor
# ez_idx = np.argmin(np.abs(ezc_loc - ms_w/2))
# ez_zloc = ezc_loc[ez_idx]
# dy_all = np.diff(ezc_loc)
# dy = dy_all[ez_idx-1] / 2 + dy_all[ez_idx] / 2
# ezc_v = ez_cross_ms[:, n_sample][ez_idx]
# dz = np.abs(ez_zloc - sub_h)

# cf = np.trapezoid(ez_cross_ms[:, n_sample][ez_idx-1:ez_idx +2], ezc_loc[ez_idx-1:ez_idx +2]) / (ez_v * 2 *cell_w)


# the hz field is very small compared to other h fields around the trace and contributes less to the overall error.
plt.figure()
plt.plot(hz_v_loc,  hz_v_ms[:, n_sample], marker=".")
plt.show()

# # fit curve using model without y offset
# y0 = ey_ms[:, n_sample][-7]
# x0 = ey_loc[-7]
# f0 = y0 * (x0 - ms_w/2) ** (1/2)
# fx = (f0 / (ey_loc - ms_w/2) ** (1/2)) 

# plt.figure()
# plt.plot(ey_loc,  ey_ms[:, n_sample], marker=".")
# plt.plot(ey_loc, fx)
# plt.show()

# y0 = ez_ms_edge[:, n_sample][-9]
# x0 = ez_loc[-9]
# f0 = y0 * (x0 - sub_h) ** (1/2)
# fx = (f0 / (ez_loc - sub_h) ** (1/2)) 

# plt.figure()
# plt.plot(ez_loc,  ez_ms_edge[:, n_sample], marker=".")
# plt.plot(ez_loc, fx)


