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

ms_w = 0.043
ms_70w = 0.023
ms_len = 0.75
ms1_y = 0

sbox_h = 0.3
sbox_w = 0.6
sbox_len = ms_len * 1.3

gap = 0.03
ms2_y = (gap / 2) + (ms_w / 2)
ms3_y = -(gap / 2) - (ms_w / 2)

sub_h = 0.02

f0 = 5e9

line_ref = rfn.elements.MSLine(h=sub_h, er=3.66, w=ms_w, length=ms_len * 1.0)
z_ref = line_ref.get_properties(f0).sel(value="z0").item()

# 70 ohm ms line on RO4350B substrate
msline70p7 = rfn.elements.MSLine(
    w=0.023, 
    h=0.020, 
    er=3.66, 
)

## get quarter wavelength at the design frequency
len_qw = msline70p7.get_wavelength(f0) / 4

# radius of curved section. Half the circumference should be len_qw
# len_qw = 2 pi r / 2
radius = len_qw.item() / np.pi

# Inner and outer radius
inner_radius = radius + (ms_70w / 2)
outer_radius = radius - (ms_70w / 2)

ms_x = (-ms_len/2, -radius)
ms2_x = (radius - 0.01, ms_len/3)

substrate = pv.Cube(center=(0, 0, sub_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sub_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

ms1_trace = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[1], ms1_y + ms_w/2, sub_h)
])

ms2_trace = pv.Rectangle([
    (ms2_x[0], ms2_y - ms_w/2, sub_h),
    (ms2_x[0], ms2_y + ms_w/2, sub_h),
    (ms2_x[1], ms2_y + ms_w/2, sub_h)
])

ms3_trace = pv.Rectangle([
    (ms2_x[0], ms3_y - ms_w/2, sub_h),
    (ms2_x[0], ms3_y + ms_w/2, sub_h),
    (ms2_x[1], ms3_y + ms_w/2, sub_h)
])

port1_face = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, 0),
])

port2_face = pv.Rectangle([
    (ms2_x[1], ms2_y - ms_w/2, sub_h),
    (ms2_x[1], ms2_y + ms_w/2, sub_h),
    (ms2_x[1], ms2_y + ms_w/2, 0),
])

port3_face = pv.Rectangle([
    (ms2_x[1], ms3_y - ms_w/2, sub_h),
    (ms2_x[1], ms3_y + ms_w/2, sub_h),
    (ms2_x[1], ms3_y + ms_w/2, 0),
])

ring = pv.Disc(
    center=(0, 0, sub_h),
    inner=inner_radius,
    outer=outer_radius,
    normal=(0, 0, 1),
    r_res=1,       # radial resolution (1 = ring)
    c_res=16       # angular resolution
)

ring = ring.clip_box((0, outer_radius + 0.1, -gap/2, gap/2, 0, sub_h)).extract_surface()

resistor = pv.Rectangle([
    (radius - 0.01, -gap/2, sub_h),
    (radius - 0.01, gap/2, sub_h),
    (radius + 0.01, gap/2, sub_h),
])

s = rfn.FDTD_Solver(sbox)
s.add_dielectric(substrate, er=3.66, loss_tan=0.002, f0=f0, style=dict(opacity=0.0))
s.add_conductor(ring, ms1_trace, ms2_trace, ms3_trace, style=dict(color="gold"))
s.add_lumped_port(1, port1_face, "z-")
s.add_lumped_port(2, port2_face, "z-")
s.add_lumped_port(3, port3_face, "z-")

# resistor.plot()

s.add_resistor(resistor, 100, "y+")


s.assign_PML_boundaries("z+", n_pml=5)

self = s
# having three cells in the PEC instead of 4 causes the edge correction to fail
# s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=4, d_sub=0.01, n_min_sub=4, blend_pec=False)
s.generate_mesh(d0 = 0.02, d_edge=0.005, z_bounds = [0.005, 0.02])
# s.edge_correction(ms1_trace)

s.add_field_monitor("mon1", "ez", "z", sub_h, 30)
# s.add_field_monitor("mon1", "ez", "z", sub_h - 0.005, 15)
s.add_field_monitor("mon2", "ey", "z", sub_h, 30)
s.add_field_monitor("mon3", "ex", "z", sub_h, 10)


# plotter = s.render()
# plotter.camera_position = "xy"
# plotter.show()


# Db_0 = s.dt / u0
# Cb_0 = s.dt / e0 
# p = s.plot_coefficients("ey_z", "a", "z", sub_h, point_size=15, cmap="brg")
# p.camera_position = "xy"
# p.show()

pulse_n = 2800
# width of half pulse in time
pulse_width = (s.dt * 400)
# center of the pulse in time
t0 = (s.dt * 500)

vsrc = 1e-2 * self.gaussian_source(width=100e-12, t0=100e-12, t_len=800e-12)
# vsrc = 1e-2 * self.gaussian_modulated_source(f0, width=600e-12, t0=200e-12, t_len=500e-12)
# plt.plot(vsrc)

frequency: np.ndarray = np.arange(1e9, 10e9, 10e6)

s.run([2], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency, source_port=2, downsample=False)
S11 = sdata[:, 0]
S21 = sdata[:, 1]
S31 = sdata[:, 2]

fig, ax = plt.subplots()
ax.plot(frequency / 1e9, conv.db20_lin(S11))
ax.plot(frequency / 1e9, conv.db20_lin(S21))
ax.plot(frequency / 1e9, conv.db20_lin(S31))
mplm.line_marker(x = f0/1e9, axes=ax, xlabel=True)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S12", "S22", "S32"])
ax.grid(True)
plt.show()


p = s.plot_monitor(["mon2"], el=0, zoom=1.1, az=0, view="xy", vmax=30, vmin=10, opacity=[0.8, 1], linear=False, cmap="jet", style="surface")
p.show(title="EM Solver")

