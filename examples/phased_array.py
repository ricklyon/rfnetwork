

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
# User Defined Parameters 
# ------------------------

f0 = 5.7e9
lam0 = rfn.const.c0_in / f0
spacing = lam0 / 2

# element positions normalized by the spacing
ele_x, ele_y = np.meshgrid([0, 1, 2, 3], [0, 1, 2, 3], indexing="ij")
ele_z = np.zeros_like(ele_x)
# Nx3 grid of element locations
ele_pos = np.array((ele_x.flatten(), ele_y.flatten(), ele_z.flatten())).T
# add coordinates
N = len(ele_pos)
ele_pos = ldarray(ele_pos * spacing, coords=dict(element=np.arange(N), axis=["x", "y", "z"]))

# solve box
pad = lam0 / 3
solve_h = conv.in_mm(20)
sbox = pv.Box((-pad, np.max(ele_x) + pad, -pad, np.max(ele_y) + pad, 0, solve_h))

# patch dimensions
sub_h = 0.039
feed_w = conv.in_mm(1.9)
feed_len = conv.in_mm(9)
inset_len = conv.in_mm(4)
inset_w = conv.in_mm(1)
patch_w = conv.in_mm(16)
patch_len = conv.in_mm(12)

# %%
#  Build Model
# ------------------------

s = rfn.FDTD_Solver(sbox)

# build patch centered at the origin
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

# combine all objects into a single patch
patch = pv.merge([feed, patch_left, patch_right, patch_top])

# create array of patches
patches = [patch.translate(p) for p in ele_pos]

# port between upper and lower leg
port_face = pv.Rectangle([
    (-feed_w / 2, -feed_len, 0),
    (feed_w / 2, -feed_len, 0),
    (feed_w / 2, -feed_len, sub_h)
])

port_faces = [port_face.translate(p) for p in ele_pos]
s.add_conductor(*patches, style=dict(color="gold", opacity=1))

# add PCB substrate
substrate = pv.Box((-pad, np.max(ele_x) + pad, -pad, np.max(ele_y) + pad, 0, sub_h))
s.add_dielectric(substrate, er=4.3, loss_tan=0.02, f0 = 5e9, style=dict(opacity=0.3))

# define lumped ports
for i, face in enumerate(port_faces):
    s.add_lumped_port(i+1, face, "z-")
    
# PML boundaries are required on all sides to add a far-field monitor
s.add_PML("y+", "y-", "x-", "x+", "z+", n_pml=3)
s.generate_mesh(d_max = 0.04, d_min=0.02)

s.add_farfield_monitor(f0)

# plot array geometry with port labels on each patch
plotter = s.render(show_mesh=False, zoom=1.5, camera_position="xy")
plotter.add_points(ele_pos, render_points_as_spheres=True, point_size=15)
plotter.add_point_labels(ele_pos, np.arange(1, N+1), always_visible=True)

fig, ax = plt.subplots()
img = plotter.screenshot()
ax.imshow(img)
ax.set_axis_off()


# %%
# Plot Array Factor
# -----------------

az = np.arange(-90, 92, 2)
el = np.arange(-90, 92, 2)

# isotropic pattern
iso_pattern = ldarray(
    np.ones((1, len(az), len(el))), coords=dict(frequency=f0, az=az, el=el)
)

# create array model from iso pattern
array_model = rfn.antennas.translate(iso_pattern, rfn.conv.m_in(ele_pos))

# compute element weights for steered beam
beam_az, beam_el = 30, 0
weights = np.conjugate(array_model.sel(az=beam_az, el=beam_el))
# use phase only 
weights = weights / np.abs(weights)
# apply weights to generate estimated beam pattern
array_factor = np.sum(array_model * weights, axis="element") / np.sqrt(N)

plt.figure()
array_factor.pcolormesh("az", "el", zfmt="db20", vmin=-25, vmax=15, cmap="jet")
plt.title(f"Array Factor, L2y, {f0/1e9:.1f}GHz")

# %%
# Solve Beam Pattern
# -----------------

# phase delay excitation
vsrc = s.gaussian_source(width=100e-12, t0=160e-12, t_len=2000e-12)
exc = [rfn.utils.phase_delay_signal(vsrc, -np.angle(w), f0=f0) for w in weights.squeeze()]
exc = ldarray(exc, coords=dict(element=np.arange(N), time=vsrc.time))

# add field monitor
s.add_field_monitor("mon1", "ey", axis="y", position=spacing, n_step=5)

# apply excitation to all elements and solve
[s.assign_excitation(exc[i], i+1) for i in range(N)]
s.solve(n_threads=4)

cpos = pv.CameraPosition(position=(1.5, -0.5, 1.3), focal_point=(1.5, 0.5, 0.3), viewup=(0, 0.0, 1.0))
fig, ax = plt.subplots(constrained_layout=True)
s.plot_monitor("mon1", opacity=1, axes=ax, camera_position=cpos, init_time=800)

# get far-field pattern in spherical coordinates
pattern = s.get_farfield_gain(theta=np.arange(0, 92, 2), phi=np.arange(-180, 182, 2))
# project polarization to L2
pattern_l2 = rfn.antennas.pattern_spherical2azel(pattern).sel(polarization="l2_el")
# transform to azel coordinate frame
beam_pattern = rfn.antennas.pattern_phitheta2azel(pattern_l2, az=az, el=el)

plt.figure()
beam_pattern.pcolormesh("az", "el", zfmt="db20", vmin=-25, vmax=15, cmap="jet")
plt.title(f"Simulated Beam Pattern, L2y, {f0/1e9:.1f}GHz")

# %%
# Plot Active S11
# -----------------
frequency: np.ndarray = np.arange(5e9, 6.4e9, 10e6)
sdata = s.get_sparameters(frequency, source_port=5)

plt.figure()
sdata.plot("frequency", yfmt="db20", xfmt=lambda x: x/1e9)
plt.legend()
plt.xlabel("Frequency [GHz]")
plt.title("Active S11")

plt.show()
