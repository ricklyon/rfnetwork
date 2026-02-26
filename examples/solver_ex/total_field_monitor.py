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

ms_w = 0.01
ms1_y = 0

sbox_h = 0.25
sbox_w = 0.4
sbox_len = 1

sub_h = 0.02
ms_x = (-sbox_len/2 + 0.1, sbox_len/2)

line = rfn.elements.MSLine(h=sub_h, er=3.66, w=ms_w)
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
    (-0.25, ms1_y - ms_w/2 - 0.001, sub_h + 0.001),
    (-0.25, ms1_y + ms_w/2 + 0.001, sub_h + 0.001),
    (-0.25, ms1_y + ms_w/2 + 0.001, sub_h - 0.001),
])


voltage_line1 = pv.Line(
    [-0.25, ms1_y, 0], [-0.25, ms1_y, sub_h]
)

voltage_line2 = pv.Line(
    [0.25, ms1_y, 0], [0.25, ms1_y, sub_h]
)

s = rfn.FDTD_Solver(sbox)
s.add_dielectric(substrate, er=3.66, style=dict(opacity=0.0))
s.add_conductor(ms1_trace, style=dict(color="gold"))
s.add_lumped_port(1, port1_face, "z-")

self = s

s.assign_PML_boundaries("x+", n_pml=10)


s.generate_mesh(d0 = 0.02, d_edge=0.005)

p1 = (ms_x[0], ms1_y + ms_w/2, sub_h)
p2 = (ms_x[1], ms1_y + ms_w/2, sub_h)
integration_line = "y+"

s.edge_correction(p1, p2, "y+")

p1 = (ms_x[0], ms1_y - ms_w/2, sub_h)
p2 = (ms_x[1], ms1_y - ms_w/2, sub_h)

s.edge_correction(p1, p2, "y-")

# total field monitor
self = s

s.add_field_monitor("mon1", "e_total", "z", sub_h, 10)

s.add_current_probe("c1", current_face)
s.add_line_probe("v1", "ez", voltage_line1)
s.add_line_probe("v2", "ez", voltage_line2)


# plotter = s.render(show_probes=True)
# plotter.camera_position = "yz"
# plotter.show()

f0 = 10e9

vsrc = 1e-2 * self.gaussian_modulated_source(f0, width=500e-12, t0=250e-12, t_len=500e-12)
plt.plot(vsrc)
frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

self = s

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]

e_tot = s.get_monitor_data("mon1")

monitor_name = "mon1"

plotter = self.plot_monitor("mon1", gif_file="fields.gif")
# plotter.show()


# z = sub_h



# # total field monitor
# x, y = np.meshgrid(ez.coords["x"], ez.coords["y"], indexing="ij")
# points = np.vstack((x.ravel(), y.ravel(), np.ones(x.size) * sub_h)).T
# # u = x / np.sqrt(x**2 + y**2)
# # v = y / np.sqrt(x**2 + y**2)
# vectors = np.vstack((ex.ravel(), ey.ravel(), ez.ravel())).T

# vectors_db = conv.db20_lin(vectors)

# mag = np.sqrt(np.sum(vectors**2, axis=1))
# mag_db = conv.db20_lin(mag)

# vectors_unit = (vectors) / mag[:, None]
# # np.sqrt(np.sum(vectors_unit**2, axis=1))


# rmin = -20
# rmax = 30
# mag_db_s = np.clip(mag_db, rmin, rmax) - rmin

# vectors_db = vectors_unit * mag_db_s[:, None]

# # scale to fit on grid
# vectors_db_grid = vectors_db * 0.01 / np.max(vectors_db)

# pdata = pv.vector_poly_data(points, vectors_db_grid)
# pdata.point_data['values'] = np.clip(mag_db, rmin, rmax).flatten(order="F")

# plotter = s.render(show_probes=False)

# arrows = pdata.glyph(orient='vectors', scale='mag')


# plotter.add_mesh(arrows, scalars="values", cmap="jet")
# plotter.show()

