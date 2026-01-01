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


s_w = 0.15
s_len = 2
s1_y = 0

sbox_w = 1
sbox_len = s_len * 1.3

sub_h = 0.06
sbox_h = sub_h * 2
s_x = ((-s_len/2), (s_len /2))

line_ref = rfn.elements.Stripline(w=s_w, b=sub_h * 2, er=1, length=s_len * 1.0)
z_ref = line_ref.get_properties(10e9).sel(value="z0").item()

substrate = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

ms1_trace = pv.Rectangle([
    (s_x[0], s1_y - s_w/2, sub_h),
    (s_x[0], s1_y + s_w/2, sub_h),
    (s_x[1], s1_y + s_w/2, sub_h)
])

port1_face = pv.Rectangle([
    (s_x[0], s1_y - s_w/2, sub_h),
    (s_x[0], s1_y + s_w/2, sub_h),
    (s_x[0], s1_y + s_w/2, 0),
])

port2_face = pv.Rectangle([
    (s_x[0], s1_y - s_w/2, sbox_h),
    (s_x[0], s1_y + s_w/2, sbox_h),
    (s_x[0], s1_y + s_w/2, sub_h),
])

port3_face = pv.Rectangle([
    (s_x[1], s1_y - s_w/2, sub_h),
    (s_x[1], s1_y + s_w/2, sub_h),
    (s_x[1], s1_y + s_w/2, 0),
])

port4_face = pv.Rectangle([
    (s_x[1], s1_y - s_w/2, sbox_h),
    (s_x[1], s1_y + s_w/2, sbox_h),
    (s_x[1], s1_y + s_w/2, sub_h),
])



s = rfn.Solver_PCB(sbox, nports=4)
s.add_substrate("sub", substrate, er=1, opacity=0.0)
s.add_pec_face("ms1", ms1_trace, color="gold")
s.add_lumped_port(1, port1_face)
s.add_lumped_port(2, port2_face)
s.add_lumped_port(3, port3_face)
s.add_lumped_port(4, port4_face)

self = s

# having three cells in the PEC instead of 4 causes the edge correction to fail
s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.02, n_min_pec=4, d_sub=0.01, n_min_sub=4, blend_pec=False)
s.init_coefficients()

s.init_ports(r0=100)
s.init_pec(edge_correction=False)
s.add_xPML(side="upper")


s.add_field_monitor("mon1", "ez", "y", 0, 5)
# s.add_field_monitor("mon1", "ey", "z", sub_h, 5)
# s.add_field_monitor("mon2", "ey", "z", sub_h, 15)
# s.add_field_monitor("mon3", "ex", "z", sub_h, 10)



# plotter = s.render(show_probes=True)
# plotter.camera_position = "yz"
# plotter.show()


Db_0 = s.dt / u0
Cb_0 = s.dt / e0 
# p = s.plot_cooeficients("hy_x", "b", "z", sub_h - 0.005, point_size=15, cmap="brg", normalization=Db_0)
# p.camera_position = "xy"
# p.show()

f0 = 10e9
pulse_n = 1200
# width of half pulse in time
t_half = (s.dt * 100)
# center of the pulse in time
t0 = (s.dt * 400)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1, 2], [vsrc, -vsrc], n_threads=4)

self = s
ports = [1, 2]

sdata = s.get_sparameters(frequency, 1, z0=100)
S11 = sdata[:, 0]
S21 = sdata[:, 2]

# # S11
# z_idx = self.ports[0]["idx"][2]
# src_component_v = self.ports[0]["values"] * (conv.m_in(self.dz[z_idx])[None, None, :, None])

# # add voltage along z
# src_voltage = -np.sum(src_component_v, axis=2)
# # average voltage across edge of port, if port is normal to y this is the x axis, if normal to x, average
# # along y
# src_axis = self.ports[0]["axis"]
# src_vp = -np.mean(src_voltage, axis=1 if src_axis == 0 else 0).squeeze() / 2
# src_applied = self.ports[0]["src"]


# As = utils.dtft(src_applied, frequency, 1 / self.dt)
# V = utils.dtft(src_vp, frequency, 1 / self.dt)
# B[:, source_port-1] = V - (As)

# # ip = -self.vi_probe_values(f"port1")

# # plt.plot(ip * 50)
# # plt.plot(-src_vp)

# Vs = utils.dtft(src_vp, frequency, 1 / self.dt)
# Is = utils.dtft(ip, frequency, 1 / self.dt)
# z0 = 50

# As = (Vs + z0 * Is) / (2 * np.sqrt(z0.real))
# Bs = (Vs - np.conj(z0) * Is) / (2 * np.sqrt(z0.real))

# S11 = Bs / As

# # S21
# vp3 = self.vi_probe_values(f"port3") * z0
# # vp4 = self.vi_probe_values(f"port4") * z0
# # h-fields are 1/2 time step ahead of the e-fields. Delay current so they are at the same time step
# Vp = utils.dtft(vp3, frequency, 1 / self.dt) * np.exp(-1j * frequency * 2 * np.pi * (self.dt / 2))


# S21 = Vp / As


# plt.plot(frequency / 1e9, conv.db20_lin(S21))



# plt.plot(a[1])
# plt.plot(a[0])

# plt.plot(v[1])
# plt.plot(a[0])
# plt.plot(S)

# plt.plot(frequency / 1e9, conv.db20_lin(S))

# sdata = s.get_sparameters(frequency)
# S11 = sdata[:, 0]


p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface")
p.camera_position = "xz"
p.show(title="EM Solver")

sdata_ref = line_ref.evaluate(frequency)["s"] 

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

fig, ax = plt.subplots()
# plt.plot(frequency / 1e9, ZP.real)
ax.plot(frequency / 1e9, conv.z_gamma(S11))
plt.ylim([0, 110])
plt.axhline(y=z_ref, linestyle=":", color="k")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
mplm.line_marker(x = 10, axes=ax)

fig.tight_layout()
plt.show()