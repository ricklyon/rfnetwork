

"""
Base Microstrip Patch Antenna
=============
"""

# sphinx_gallery_thumbnail_number = -1

import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv
import pyvista as pv
from np_struct import ldarray

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
ele_x, ele_y = np.meshgrid([0, 1, 2, 3], [0, 1, 2, 3], indexing="ij")
ele_z = np.zeros_like(ele_x)
# Nx3 grid of element locations
ele_pos = np.array((ele_x.flatten(), ele_y.flatten(), ele_z.flatten())).T
# add coordinates
N = len(ele_pos)
ele_pos = ldarray(ele_pos, coords=dict(element=np.arange(N), axis=["x", "y", "z"]))

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

s.add_farfield_monitor(f0)

plotter = s.render(show_mesh=True)
plotter.show()


# %%
# Solve Embedded Element Pattern
# ------------------------

# apply excitation to center element
vsrc = s.gaussian_source(width=100e-12, t0=160e-12, t_len=2000e-12)
s.assign_excitation(vsrc, 5)

s.solve(n_threads=4)

frequency: np.ndarray = np.arange(5e9, 6.4e9, 10e6)
sdata = s.get_sparameters(frequency, downsample=False, source_port=5)

plt.figure()
sdata.plot("frequency", yfmt="db20")
plt.legend()

def get_pattern():
    pattern = s.get_farfield_gain(theta=np.arange(0, 92, 2), phi=np.arange(-180, 182, 2))

    pattern_l2 = rfn.antennas.pattern_spherical2azel(pattern).sel(polarization="l2_el")

    return rfn.antennas.pattern_phitheta2azel(pattern_l2, az=np.arange(-90, 92, 2), el=np.arange(-90, 92, 2))

plt.figure()
emb_pattern = get_pattern()
emb_pattern.pcolormesh("az", "el", zfmt="db20", vmin=-20, cmap="jet")


# %%
# Estimate Beam Pattern
# -----------------

# create array model from embedded element pattern
array_model = rfn.antennas.translate(emb_pattern, rfn.conv.m_in(ele_pos))

# compute element weights for steered beam
beam_az, beam_el = 30, 20
weights = np.conjugate(array_model.sel(az=beam_az, el=beam_el))
# use phase only 
weights = weights / np.abs(weights)
# apply weights to generate estimated beam pattern
beam_pattern_estimated = np.sum(array_model * weights, axis="element") / np.sqrt(N)

plt.figure()
beam_pattern_estimated.pcolormesh("az", "el", zfmt="db20", vmin=-25, vmax=10, cmap="jet")
plt.title(f"Estimated Beam Pattern, L2y, {f0/1e9:.1f}GHz")

# %%
# Solve Beam Pattern
# -----------------

# phase delay excitation
exc = [rfn.utils.phase_delay_signal(vsrc, -np.angle(w), f0=f0) for w in weights.squeeze()]
exc = ldarray(exc, coords=dict(element=np.arange(N), time=vsrc.time))

# plot excitations for all elements
fig = plt.figure()
exc.plot("time")

# apply excitation to all elements and solve
[s.assign_excitation(exc[i], i+1) for i in range(N)]
s.solve(n_threads=4)

frequency: np.ndarray = np.arange(5e9, 6.4e9, 10e6)
sdata = s.get_sparameters(frequency, downsample=False, source_port=5)

plt.figure()
sdata.plot("frequency", yfmt="db20")

beam_pattern = get_pattern()
plt.figure()
beam_pattern.pcolormesh("az", "el", zfmt="db20", vmin=-25, vmax=10, cmap="jet")
plt.title(f"Simulated Beam Pattern, L2y, {f0/1e9:.1f}GHz")


