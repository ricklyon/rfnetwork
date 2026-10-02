

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

import pyvista as pv

# pv.set_jupyter_backend("trame")

# %%
# User defined Parameters [inches]
# ------------------------

f0 = 5.8e9
lam0 = rfn.const.c0_in / f0
spacing = lam0 / 2

# element positions normalized by the spacing
ele_x, ele_y = np.meshgrid([0, 1, 2], [0, 1, 2], indexing="ij")
ele_z = np.zeros_like(ele_x)

# solve box
pad = lam0 / 3
solve_h = conv.in_mm(10)
sbox = pv.Box((-pad, np.max(ele_x) + pad, -pad, np.max(ele_y) + pad, 0, solve_h))

sub_h = 0.039
feed_w = conv.in_mm(1.9)
feed_len = conv.in_mm(9)
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

patch = pv.merge([feed, patch_left, patch_right, patch_top])

# Nx3 grid of element locations
ele_pos = np.array((ele_x.flatten(), ele_y.flatten(), ele_z.flatten())).T

patches = [patch.translate(p) for p in ele_pos]

substrate = pv.Box((-pad, np.max(ele_x) + pad, -pad, np.max(ele_y) + pad, 0, sub_h))

# port between upper and lower leg
port_face = pv.Rectangle([
    (-feed_w / 2, -feed_len, 0),
    (feed_w / 2, -feed_len, 0),
    (feed_w / 2, -feed_len, sub_h)
])

port_faces = [port_face.translate(p) for p in ele_pos]


s.add_conductor(*patches, style=dict(color="gold", opacity=1))
s.add_dielectric(substrate, er=4.3, loss_tan=0.02, f0 = 5e9, style=dict(opacity=0.3))

for i, face in enumerate(port_faces):
    s.add_lumped_port(i+1, face, "z-")
    
# PML boundaries are required on all sides to add a far-field monitor
s.add_PML("y+", "y-", "x-", "x+", "z+", n_pml=3)
s.generate_mesh(d_max = 0.03, d_min=0.01)


plotter = s.render(show_mesh=True)
plotter.show()


# %%
# Setup Excitation and Solve
# ------------------------

s.add_field_monitor("mon1", "ez", axis="z", position=sub_h, n_step=10)

vsrc = s.gaussian_source(width=100e-12, t0=60e-12, t_len=2000e-12)
[s.assign_excitation(vsrc, i) for i in range(1, len(ele_pos) +1)]

print("n cells: ", s.Nx, s.Ny, s.Nz)
print("time steps: ", len(s.time))

s.add_farfield_monitor(f0)

s.solve(n_threads=4)

frequency: np.ndarray = np.arange(5e9, 6.4e9, 10e6)
sdata = s.get_sparameters(frequency, downsample=False, source_port=5)

plt.figure()
sdata.plot("frequency", yfmt="db20", b=[1, 4])


pattern = s.get_farfield_gain(theta=np.arange(0, 92, 2), phi=np.arange(-180, 182, 2))

pattern_l2 = rfn.antennas.pattern_spherical2azel(pattern).sel(polarization="l2_el")

pattern_azel = rfn.antennas.pattern_phitheta2azel(pattern_l2, az=np.arange(-90, 92, 2), el=np.arange(-90, 92, 2))

plt.figure()
pattern_azel.pcolormesh("az", "el", zfmt="db20", vmin=-30, cmap="jet")

plt.figure()
pattern_azel.plot("el", az=0, yfmt="db20", ymin=-20)


# calculate memory usage

