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

sbox_h = 0.5
sbox_w = 0.6
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

via = pv.Box(bounds=(ms_x[1], ms_x[1] + 0.02, -0.02, 0.02, 0, sub_h))


voltage_line = pv.Line(
    [0, ms1_y, 0], [0, ms1_y, sub_h]
)

s = rfn.EM_Solver(sbox)
s.add_dielectric("sub", substrate, er=3.66, style=dict(opacity=0.0))
s.add_conductor("ms1", ms1_trace, style=dict(color="gold"))
s.add_conductor("via", via, style=dict(color="gold", opacity=1))
s.add_lumped_port(1, port1_face)
# s.add_lumped_port(2, port2_face)

s.assign_PML_boundaries("z+", "y-", "y+", n_pml=5)

self = s
obj = via
# having three cells in the PEC instead of 4 causes the edge correction to fail
# s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=4, d_sub=0.01, n_min_sub=4, blend_pec=False)
s.generate_mesh(d0 = 0.02, d_edge=0.005, z_bounds = [0.005, 0.02])
s.edge_correction(ms1_trace)

s.add_field_monitor("mon1", "ez", "z", sub_h, 20)
# s.add_field_monitor("mon1", "ez", "z", sub_h - 0.005, 15)
s.add_field_monitor("mon2", "ez", "y", 0, 5)
s.add_field_monitor("mon3", "ex", "z", sub_h, 10)

s.add_current_probe("c1", current_face)
s.add_line_probe("v1", "ez", voltage_line)


plotter = s.render(show_probes=True)
plotter.camera_position = "xy"
plotter.show()

# Db_0 = s.dt / u0
# Cb_0 = s.dt / e0 
# p = s.plot_coefficients("ex_y", "a", "z", sub_h, point_size=15, cmap="brg")
# p.camera_position = "xy"
# p.show()

###########################

obj = ms1_trace
points = (0, 0, 0.02)

    


###########################
f0 = 10e9
pulse_n = 2800
# width of half pulse in time
pulse_width = (s.dt * 400)
# center of the pulse in time
t0 = (s.dt * 500)

vsrc = 1e-2 * self.gaussian_source(width=80e-12, t0=80e-12, t_len=500e-12)
plt.plot(vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency, downsample=False)
S11 = sdata[:, 0]
# S21 = sdata[:, 1]


p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", vmin=-10, vmax=10, opacity=[0.8, 1], linear=True, cmap="RdBu", style="surface",)
p.show(title="EM Solver")

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v = self.vi_probe_values("v1")

IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(-self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
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
# mplm.line_marker(x = 10, axes=ax)
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




fig.tight_layout()
plt.show()
