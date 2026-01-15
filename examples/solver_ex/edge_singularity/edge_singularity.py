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

sbox_h = 0.25
sbox_w = 0.4
sbox_len = 1

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

voltage_line2 = pv.Line(
    [0.25, ms1_y, 0], [0.25, ms1_y, sub_h]
)

s = rfn.Solver_PCB(sbox, nports=1)
s.add_substrate("sub", substrate, er=er, opacity=0.0)
s.add_pec_face("ms1", ms1_trace, color="gold")
s.add_lumped_port(1, port1_face)

self = s

# d0 = 0.01
# d_pec = 0.01
# n_min_pec=4
# d_sub=0.01
# n_min_sub=4
# n0 = 2

run_ref = False

if run_ref:
    d_pec = 0.0025 # 0.01 # 0.0025 
else:
    d_pec = 0.01

s.init_mesh(d0 = 0.01, n0 = 3, d_pec = d_pec, n_min_pec=3, d_sub=0.005, n_min_sub=8, blend_pec=False)
s.init_coefficients()

s.init_ports()
s.add_xPML(side="upper")

s.init_pec(hy_CF=3, hx_CF=3, hz_CF=1/3)

s.add_field_monitor("hz", "hz", "z", sub_h, 1)
s.add_field_monitor("ey", "ey", "z", sub_h, 1)
s.add_field_monitor("ex", "ex", "z", sub_h, 1)

s.add_current_probe("c1", current_face)
s.add_voltage_probe("v1", voltage_line1)
s.add_voltage_probe("v2", voltage_line2)


plotter = s.render(show_probes=True)
plotter.camera_position = "yz"
plotter.show()


Db_0 = s.dt / u0
Cb_0 = s.dt / e0 
# p = s.plot_cooeficients("ex_z", "b", "z", sub_h, point_size=15, cmap="brg", normalization=Cb_0)
# p.camera_position = "xy"
# p.show()
p = s.plot_cooeficients("hy_x1", "b", "x", 0, point_size=15, cmap="brg", normalization=Db_0)
p.camera_position = "yz"
p.show()



vsrc = 1e-2 * self.gaussian_source(width=50e-12, t_len=130e-12)
t = np.linspace(0, self.dt * len(vsrc), len(vsrc))
# plt.plot(t / 1e-12, vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]


p = s.plot_monitor(["ey"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="points")
p.show(title="EM Solver")

# p = s.plot_monitor(["mon3"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="points")
# p.show(title="EM Solver")

t_sample = 72e-12

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v1 = self.vi_probe_values("v1")
line_v2 = self.vi_probe_values("v2")

if run_ref:
    np.save("line_v1_ref", line_v1)
    np.save("line_i1_ref", line_i)
    np.save("ref_time", t)

line_i_ref = np.load(dir_ / "line_i1_ref.npy")
line_v1_ref = np.load(dir_ / "line_v1_ref.npy")
t_ref = np.load(dir_ / "ref_time.npy")

plt.figure()
plt.plot(t_ref, line_v1_ref)
plt.plot(t, line_v1)
mplm.line_marker(x=t_sample)

plt.figure()
plt.plot(t_ref, line_i_ref)
plt.plot(t, line_i)
mplm.line_marker(x=t_sample)



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

vp_e = get_vp(line_v1, line_v2, conv.m_in(0.25))

print("vp / c", vp_e / rfn.const.c0)



def mon_yslice(monitor, t_sample):
    n_sample = int(t_sample / self.dt)
    x_len = self.monitors[monitor]["shape"][0]
    yloc = self.floc[monitor][1]
    return yloc, self.monitors[monitor]["values"][n_sample, int(x_len / 2)]



IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
ZP = VP / IP

fig, ax = plt.subplots()
plt.plot(frequency / 1e9, ZP.real)
ax.plot(frequency / 1e9, conv.z_gamma(S11))
plt.ylim([0, 120])
plt.axhline(y=z_ref, linestyle=":", color="k")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
mplm.line_marker(x = 10, axes=ax)

S11_z = conv.gamma_z(ZP)

# hz and ey components along y

hz_yloc, hz_values = mon_yslice("hz", t_sample = t_sample)
ey_yloc, ey_values = mon_yslice("ey", t_sample = t_sample)

# d_pec = 0.0025
if run_ref:
    np.save("hz_fine_grid", hz_values)
    np.save("hz_loc_fine_grid", hz_yloc)
    np.save("ey_fine_grid", ey_values)
    np.save("ey_loc_fine_grid", ey_yloc)

hz_ref = np.load(dir_ / "hz_fine_grid.npy")
hz_ref_loc = np.load(dir_ / "hz_loc_fine_grid.npy")
ey_ref = np.load(dir_ / "ey_fine_grid.npy")
ey_ref_loc = np.load(dir_ / "ey_loc_fine_grid.npy")


# ez components along z below the trace
n_sample = int(t_sample / self.dt)
name = "v1"
ez_values = np.array([p["values"][n_sample] for k, p in self.probes.items() if k[:len(name)] == name])
ez_zloc = np.array([self.floc["ez"][2][p["index"][2]] for k, p in self.probes.items() if k[:len(name)] == name])

if run_ref:
    np.save("ez_fine_grid", ez_values)
    np.save("ez_loc_fine_grid", ez_zloc)

ez_ref = np.load(dir_ / "ez_fine_grid.npy")
ez_ref_loc = np.load(dir_ / "ez_loc_fine_grid.npy")

# hy components above the trace
self.probes["c1_2"]
name = "c1"
hy_values = np.array([p["values"][n_sample] for k, p in self.probes.items() if k[:len(name)] == name])[2:]
hy_ylocs = np.array([self.floc["hy"][1][p["index"][1]] for k, p in self.probes.items() if k[:len(name)] == name])[2:]

if run_ref:
    np.save("hy_fine_grid", hy_values)
    np.save("hy_loc_fine_grid", hy_ylocs)

hy_ref = np.load(dir_ / "hy_fine_grid.npy")
hy_ref_loc = np.load(dir_ / "hy_loc_fine_grid.npy")

# plt.figure()
# plt.plot(hy_ref_loc[::2], hy_ref[::2] * conv.m_in(0.0025))
# plt.plot(hy_ylocs[::2], hy_values[::2] * conv.m_in(0.01))

plt.figure()
plt.plot(hy_ref_loc[::2], hy_ref[::2], marker=".")
plt.plot(hy_ylocs[::2], hy_values[::2], marker=".")
plt.xlabel("y [in]")
plt.ylabel("Hy")
plt.title("Hy Below Trace")

plt.figure()
plt.plot(hy_ref_loc[1::2], hy_ref[1::2], marker=".")
plt.plot(hy_ylocs[1::2], hy_values[1::2], marker=".")
plt.xlabel("y [in]")
plt.ylabel("Hy")
plt.title("Hy Above Trace")


ez_ref = np.load(dir_ / "ez_fine_grid.npy")
ez_ref_loc = np.load(dir_ / "ez_loc_fine_grid.npy")

plt.figure()
plt.plot(ez_ref_loc, ez_ref, marker=".")
plt.plot(ez_zloc, ez_values, marker=".")
plt.xlabel("z [in]")
plt.ylabel("Ez")

# hz and ey in the plane of the trace along y
# plt.figure()

# plt.plot(hz_ref_loc, hz_ref, marker=".")
# plt.plot(hz_yloc, hz_values, marker=".")
# plt.xlabel("y [in]")
# plt.ylabel("Hz")

# plt.figure()
# plt.plot(ey_ref_loc, ey_ref, marker=".")
# plt.plot(ey_yloc, ey_values, marker=".")
# plt.xlabel("y [in]")
# plt.ylabel("Ey")


# fig, ax = plt.subplots()
# rfn.plots.draw_smithchart(ax)
# plt.plot(S11.real, S11.imag)
# plt.plot(S11_z.real, S11_z.imag)

# fig, ax = plt.subplots()
# plt.plot(frequency, conv.db20_lin(S11))
plt.show()