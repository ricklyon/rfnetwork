import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time
from np_struct import ldarray

import rfnetwork as rfn
import mpl_markers as mplm
import sys
from pathlib import Path

dir_ = Path(__file__).parent

DATA_DIR = dir_ / "data"


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

sbox_h = 0.25
sbox_w = 0.4
sbox_len = 0.5

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

current_face = pv.Rectangle([
    (0, ms1_y - ms_w/2 - 0.001, sub_h + 0.001),
    (0, ms1_y + ms_w/2 + 0.001, sub_h + 0.001),
    (0, ms1_y + ms_w/2 + 0.001, sub_h - 0.001),
])


voltage_line1 = pv.Line(
    [0, ms1_y, 0], [0, ms1_y, sub_h]
)


s = rfn.Solver_3D(sbox)
s.add_dielectric("sub", substrate, er=er, style=dict(opacity=0.0))
s.add_conductor("ms1", ms1_trace, style=dict(color="gold"))
s.add_lumped_port(1, port1_face)


self = s

s.assign_PML_boundaries("x+", "z+", "y-", "y+", n_pml=5)

s.generate_mesh(d0=0.01, d_edge=0.005, z_bounds=[0.005, 0.01])

pec_face = ms1_trace
s.edge_correction(ms1_trace)

s.add_current_probe("c1", current_face)
s.add_line_probe("v1", "ez", voltage_line1)

s.add_field_monitor("ex", "ex", "x", 0, 10)
s.add_field_monitor("ey", "ey", "x", 0, 10)
s.add_field_monitor("ez", "ez", "x", 0, 10)
s.add_field_monitor("hx", "hx", "x", 0, 10)
s.add_field_monitor("hy", "hy", "x", 0, 10)
s.add_field_monitor("hz", "hz", "x", 0, 10)

s.add_field_monitor("ez_xy", "ez", "z", 0, 10)

Db_0 = s.dt / u0
p = s.plot_coefficients("hy_z1", "b", "z", sub_h, point_size=15, cmap="brg", normalization=Db_0, vmax=1.5)
p.camera_position = "xy"
p.show()

# p = s.plot_coefficients("hy_z1", "b", "z", sub_h, point_size=15, cmap="brg")
# p.camera_position = "xy"
# p.show()


# plotter = s.render(show_probes=True)
# plotter.camera_position = "yz"
# plotter.show()
# print(s.Nx * s.Ny * s.Nz / 1e3)

t_len=130e-12
vsrc = 1e-2 * self.gaussian_source(width=30e-12, t0 = 37e-12, t_len=t_len)
t = np.linspace(0, self.dt * len(vsrc), len(vsrc))
# plt.plot(t / 1e-12, vsrc)


frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]


p = s.plot_monitor(["ez_xy"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface", vmax=20)
p.show(title="EM Solver")

# edge correction
pec_face = ms1_trace

# sample the x=0 fields when they are at their strongest
t0 = 56e-12
ex_plane = self.get_monitor_data("ex").sel(time=t0)
ey_plane = self.get_monitor_data("ey").sel(time=t0)
ez_plane = self.get_monitor_data("ez").sel(time=t0)

hx_plane = self.get_monitor_data("hx").sel(time=t0)
hy_plane = self.get_monitor_data("hy").sel(time=t0)
hz_plane = self.get_monitor_data("hz").sel(time=t0)


# E fields
plt.plot(ey_plane.coords["y"], ey_plane.sel(z=sub_h), marker=".")
plt.plot(ez_plane.coords["z"], ez_plane.sel(y=ms_w/2), marker=".")

plt.plot(ey_plane.coords["z"], ey_plane.sel(y=ms_w/2 +0.005), marker=".")
plt.plot(ez_plane.coords["y"], ez_plane.sel(z=sub_h - 0.005), marker=".")

# H fields
plt.plot(hy_plane.coords["z"], hy_plane.sel(y=ms_w/2), marker=".")
plt.plot(hz_plane.coords["y"], hz_plane.sel(z=sub_h), marker=".")

plt.plot(hy_plane.coords["y"], hy_plane.sel(z=sub_h - 0.005), marker=".")
plt.plot(hz_plane.coords["z"], hz_plane.sel(y=ms_w/2 + 0.005), marker=".")


# x components do not contribute much error relative to the singularities in the other components
# plt.plot(ex_plane.coords["y"], ex_plane.sel(z=sub_h), marker=".")
# plt.plot(ex_plane.coords["z"], ex_plane.sel(y=ms_w/2), marker=".")

# plt.plot(hx_plane.coords["y"], hx_plane.sel(z=sub_h), marker=".")
# plt.plot(hx_plane.coords["z"], hx_plane.sel(y=ms_w/2 + 0.001), marker=".")



IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(-self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
ZP = VP / IP

fig, ax = plt.subplots()
plt.plot(frequency / 1e9, ZP.real)
plt.plot(frequency / 1e9, conv.db20_lin(S11))
plt.ylim([0, 120])
plt.axhline(y=z_ref, linestyle=":", color="k")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
mplm.line_marker(x = 10, axes=ax)

