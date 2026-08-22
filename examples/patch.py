

"""
Base Microstrip Patch Antenna
=============
"""

# sphinx_gallery_thumbnail_number = -1

import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv
import pyvista as pv

import rfnetwork as rfn
import mpl_markers as mplm
from rfnetwork.core.units import conv

# set matplotlib style
plt.style.use(rfn.DEFAULT_STYLE)

# %%
# User defined Parameters [inches]
# ------------------------

# solve box
solve_w = conv.in_mm(35)
solve_len = conv.in_mm(38)
solve_h = conv.in_mm(10)
sbox = pv.Cube(center=(0, 0, solve_h / 2), x_length=solve_w, y_length=solve_len, z_length=solve_h)

sub_h = 0.039
feed_w = conv.in_mm(1.9)
feed_len = conv.in_mm(9) # from center
inset_len = conv.in_mm(4)
inset_w = conv.in_mm(1)
patch_w = conv.in_mm(16)
patch_len = conv.in_mm(12)


s = rfn.FDTD_Solver(sbox)
feed = pv.Rectangle([(-feed_w / 2, 0, sub_h), (feed_w / 2, 0, sub_h), (feed_w / 2, -feed_len, sub_h)])

patch_left = pv.Rectangle(
    [(-patch_w / 2, -patch_len / 2, sub_h), (-patch_w / 2, patch_len / 2, sub_h), (-inset_w / 2 - feed_w / 2, patch_len / 2, sub_h)]
)

patch_right = pv.Rectangle(
    [(patch_w / 2, -patch_len / 2, sub_h), (patch_w / 2, patch_len / 2, sub_h), (inset_w / 2 + feed_w / 2, patch_len / 2, sub_h)]
)

patch_top = pv.Rectangle(
    [(-patch_w / 2, -patch_len / 2 + inset_len, sub_h), (-patch_w / 2, patch_len / 2, sub_h), (patch_w / 2, patch_len / 2, sub_h)]
)

substrate = pv.Cube(center=(0, 0, sub_h / 2), x_length=solve_w, y_length=solve_len, z_length=sub_h)

s.add_conductor(feed, patch_left, patch_right, patch_top, style=dict(color="gold"))
s.add_dielectric(substrate, er=4.3, loss_tan=0.02, f0 = 5e9, style=dict(opacity=0.5))

# port between upper and lower leg
port1_face = pv.Rectangle([
    (-feed_w / 2, -feed_len, 0),
    (feed_w / 2, -feed_len, 0),
    (feed_w / 2, -feed_len, sub_h)
])


s.add_lumped_port(1, port1_face, "z-")



# PML boundaries are required on all sides to add a far-field monitor
s.assign_PML_boundaries("x-", "x+", "y-", "y+", "z+", n_pml=5)
s.generate_mesh(d_max = 0.015, d_min=0.005)

plotter = s.render(show_mesh=True)
# plotter.show()

self = s

# plotter.show()

# s.plot_coefficients("ey_x", "a", "z", position=sub_h, point_size=15, cmap="brg").show()

# %%
# Setup Excitation and Solve
# ------------------------

# s.add_field_monitor("mon1", "ez", axis="z", position=sub_h, n_step=10)

vsrc = s.gaussian_source(width=100e-12, t0=60e-12, t_len=2000e-12)
s.assign_excitation(vsrc, 1)

print("n cells: ", s.Nx, s.Ny, s.Nz)
print("time steps: ", len(s.time))


s.solve(n_threads=4)

# p = s.plot_monitor(
#     "mon1", camera_position="xy", vmin=20, vmax=60, opacity=1
# )
# p.show()

frequency: np.ndarray = np.arange(5e9, 6.4e9, 1e6)
sdata_raw = s.get_sparameters(frequency, downsample=False)
# cast as component to use plot functions
sdata = rfn.Component_Data(sdata_raw)

sdata.plot(11, fmt="db")
plt.show()

# calculate memory usage

