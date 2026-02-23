import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time

import rfnetwork as rfn
import mpl_markers as mplm
import sys
import itertools

from scipy.spatial.transform import Rotation

pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)

sys.argv = sys.argv[0:1]

u0 = const.u0
e0 = const.e0
c0 = const.c0

ms_w = 0.04
ms_len0 = 0.15

# angled section length
ms_ang = 45
ms_len1 = 0.4
ms_len1_x = ms_len1 * np.cos(np.deg2rad(ms_ang))
ms_len1_y = ms_len1 * np.sin(np.deg2rad(ms_ang))

sbox_h = 0.35
sbox_w = ms_len1_x + 0.5
sbox_len = ms_len1_y + 0.5

sub_h = 0.02
ms_x = ((-ms_len1_x / 2), (ms_len1_x / 2))
ms_y = (-ms_len1_y / 2, ms_len1_y / 2)


substrate = pv.Cube(center=(0, 0, sub_h/2), x_length=sbox_w, y_length=sbox_len, z_length=sub_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)

ms0_trace = pv.Rectangle([
    (ms_x[0], ms_y[0] - ms_w/2, sub_h),
    (ms_x[0], ms_y[0] + ms_w/2, sub_h),
    (ms_x[0] - ms_len0, ms_y[0] + ms_w/2, sub_h)
])

ms1_trace = pv.Rectangle([
    (ms_x[1], ms_y[1] - ms_w/2, sub_h),
    (ms_x[1], ms_y[1] + ms_w/2, sub_h),
    (ms_x[1] + ms_len0, ms_y[1] + ms_w/2, sub_h)
])

ms_ang_trace = pv.Rectangle([
    (0, - ms_w/2, sub_h),
    (0, ms_w/2, sub_h),
    (ms_len1, ms_w/2, sub_h)
])

# rotate trace
rot = Rotation.from_euler("z", ms_ang, degrees=True).as_matrix()
ms_ang_trace.points = np.einsum("ij,kj->ki", rot, ms_ang_trace.points)
# translate to position
ms_ang_trace.points += (ms_x[0], ms_y[0], 0)

# corners
corner0 = pv.Triangle([ms_ang_trace.points[0], (ms_x[0], ms_y[0], sub_h), (ms_x[0], ms_y[0] - ms_w/2, sub_h)])

corner1 = pv.Triangle([ms_ang_trace.points[2], (ms_x[1], ms_y[1], sub_h), (ms_x[1], ms_y[1] + ms_w/2, sub_h)])


port1_face = pv.Rectangle([
    (ms_x[0] - ms_len0, ms_y[0] - ms_w/2, sub_h),
    (ms_x[0] - ms_len0, ms_y[0] + ms_w/2, sub_h),
    (ms_x[0] - ms_len0, ms_y[0] + ms_w/2, 0),
])

port2_face = pv.Rectangle([
    (ms_x[1] + ms_len0, ms_y[1] - ms_w/2, sub_h),
    (ms_x[1] + ms_len0, ms_y[1] + ms_w/2, sub_h),
    (ms_x[1] + ms_len0, ms_y[1] + ms_w/2, 0),
])


s = rfn.FDTD_Solver(sbox)
s.add_dielectric(substrate, er=3.66, style=dict(opacity=0.0))
s.add_conductor(ms0_trace, style=dict(color="gold"))
s.add_conductor(ms1_trace, style=dict(color="gold"))
s.add_conductor(ms_ang_trace, style=dict(color="gold"))
s.add_conductor(corner0, corner1, style=dict(color="gold"))
s.add_lumped_port(1, port1_face, "z-")
s.add_lumped_port(2, port2_face, "z-")

# test_box = pv.Box((-0.1, 0.1, -0.1, 0.1, 0, 0.1))
# s.add_conductor("test", test_box, style=dict(color="gold", opacity=0.5))

s.assign_PML_boundaries("z+", "y-", "y+", n_pml=5)

self = s

# angled edge mesh generation
d0 = 0.02
d_edge = 0.005


# having three cells in the PEC instead of 4 causes the edge correction to fail
# s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=4, d_sub=0.01, n_min_sub=4, blend_pec=False)
s.generate_mesh(d0 = 0.02, d_edge=0.005, z_bounds = [0.005, 0.02])

plotter = s.render()
plotter.camera_position = "xy"
plotter.show()
print(s.Nx * s.Ny * s.Nz / 1e3)

obj = ms_ang_trace
x= 0
y = 0



s.add_field_monitor("mon1", "ez", "z", sub_h, 15)
# s.add_field_monitor("mon1", "ez", "z", sub_h - 0.005, 15)
s.add_field_monitor("mon2", "ez", "y", 0, 5)
s.add_field_monitor("mon3", "ex", "z", sub_h, 10)




Db_0 = s.dt / u0
Cb_0 = s.dt / e0 
p = s.plot_coefficients("ex_y", "b", "z", sub_h, point_size=15, cmap="brg")
p.camera_position = "xy"
p.show()

f0 = 10e9
vsrc = 1e-2 * self.gaussian_modulated_source(f0, width=200e-12, t0=120e-12, t_len=380e-12)
plt.plot(vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency, downsample=False)
S11 = sdata[:, 0]
S21 = sdata[:, 1]


p = s.plot_monitor(
    ["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface",
)
p.show(title="EM Solver")

# sdata_ref = line_ref.evaluate(frequency)["s"] 



fig, axes = plt.subplots(2, 2, figsize=(9, 9))

ax = axes[0,0]
rfn.plots.draw_smithchart(ax)
ax.plot(S11.real, S11.imag)
# ax.plot(sdata_ref.sel(b=1, a=1).real, sdata_ref.sel(b=1, a=1).imag)

ax = axes[0,1]
ax.plot(frequency / 1e9, conv.db20_lin(S11))
# ax.plot(frequency / 1e9, conv.db20_lin(sdata_ref).sel(b=1, a=1))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S11", "Ref"])

ax = axes[1,0]
ax.plot(frequency / 1e9, conv.db20_lin(S21))
# ax.plot(frequency / 1e9, conv.db20_lin(sdata_ref).sel(b=2, a=1))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S21", "Ref"])

ax = axes[1,1]
ax.plot(frequency / 1e9, np.unwrap(np.angle(S21, deg=True)))
# ax.plot(frequency / 1e9, np.unwrap(np.angle(sdata_ref.sel(b=2, a=1), deg=True)))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[deg]")
ax.legend(["S21", "Ref"])


fig.tight_layout()
plt.show()
