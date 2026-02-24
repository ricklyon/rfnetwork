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

sbox_h = 0.5
sbox_w = 0.6
sbox_len = ms_len * 1.3

sub_h = 0.02
ms_ends = ((-ms_len/2), sbox_len/2)

line_ref = rfn.elements.MSLine(h=sub_h, er=3.66, w=ms_w, length=ms_len * 1.0)
z_ref = line_ref.get_properties(10e9).sel(value="z0").item()

# len_axis = 2
# width_axis = 1
# normal_axis = 0

len_axis = 0
width_axis = 1
normal_axis = 2

def build_dims(len, width, height):
    dimensions = [None for i in range(3)]

    dimensions[len_axis] = len
    dimensions[width_axis] = width
    dimensions[normal_axis] = height

    return dimensions


sub_size = build_dims(sbox_len, sbox_w, sub_h)
substrate = pv.Cube(
    center=build_dims(0, 0, sub_h/2), 
    x_length=sub_size[0], y_length=sub_size[1], z_length=sub_size[2]
)

sbox_size = build_dims(sbox_len, sbox_w, sbox_h)
sbox = pv.Cube(
    center=build_dims(0, 0, sbox_h/2), 
    x_length=sbox_size[0], y_length=sbox_size[1], z_length=sbox_size[2]
)

ms1_trace = pv.Rectangle([
    build_dims(ms_ends[0], - ms_w/2, sub_h),
    build_dims(ms_ends[0], + ms_w/2, sub_h),
    build_dims(ms_ends[1], + ms_w/2, sub_h)
])

port1_face = pv.Rectangle([
    build_dims(ms_ends[0], - ms_w/2, sub_h),
    build_dims(ms_ends[0], + ms_w/2, sub_h),
    build_dims(ms_ends[0], + ms_w/2, 0),
])

current_face = pv.Rectangle([
    build_dims(0, - ms_w/2 - 0.001, sub_h + 0.001),
    build_dims(0, + ms_w/2 + 0.001, sub_h + 0.001),
    build_dims(0, + ms_w/2 + 0.001, sub_h - 0.001),
])


voltage_line = pv.Line(
    build_dims(0, 0, sub_h), build_dims(0, 0, 0)
)

s = rfn.FDTD_Solver(sbox)
s.add_dielectric(substrate, er=3.66, style=dict(opacity=0.0))
s.add_conductor(ms1_trace, style=dict(color="gold"))

int_axis = ["x-", "y-", "z-"][normal_axis]
s.add_lumped_port(1, port1_face, integration_axis=int_axis)

pml_side = ["x", "y", "z"][len_axis]
s.assign_PML_boundaries(f"{pml_side}+", n_pml=10)

self = s

s.generate_mesh(d0 = 0.02)

wa = ["x", "y", "z"][width_axis]
p1 = build_dims(ms_ends[0], + ms_w/2, sub_h)
p2 = build_dims(ms_ends[1], + ms_w/2, sub_h)

s.edge_correction(p1, p2, f"{wa}+")

p1 = build_dims(ms_ends[0], - ms_w/2, sub_h)
p2 = build_dims(ms_ends[1], - ms_w/2, sub_h)

s.edge_correction(p1, p2, f"{wa}-")

# efield normal to trace
e_normal = f"e{int_axis[0]}"
s.add_field_monitor("mon1", e_normal, e_normal[1], sub_h, 5)

s.add_current_probe("c1", current_face)
s.add_line_probe("v1", e_normal, voltage_line)

plotter = s.render(show_probes=True)
plotter.show()


# Db_0 = s.dt / u0
# Cb_0 = s.dt / e0 
p = s.plot_coefficients(f"{e_normal}_{pml_side[0]}", "a", f"{pml_side[0]}", ms_ends[0], point_size=15, cmap="brg")
p.camera_position = "xy"
p.show()


vsrc = 1e-2 * self.gaussian_source(width=80e-12, t0=80e-12, t_len=300e-12)
# plt.plot(vsrc)
frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)


p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface")
p.show(title="EM Solver")


sdata = s.get_sparameters(frequency, downsample=False)
S11 = sdata[:, 0]



# compute line impedance
line_i = self.vi_probe_values("c1")
line_v = self.vi_probe_values("v1")

IP = utils.dtft(-self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt)
ZP = VP / IP
S11_zp = rfn.conv.gamma_z(ZP)

fig, ax = plt.subplots()
plt.plot(frequency / 1e9, ZP.real)
ax.plot(frequency / 1e9, conv.z_gamma(S11))
plt.ylim([0, 120])
plt.axhline(y=z_ref, linestyle=":", color="k")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
mplm.line_marker(x = 10, axes=ax)
ax.legend(["Probe", "Port"])



fig.tight_layout()
plt.show()
