import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time

import rfnetwork as rfn
import mpl_markers as mplm
import sys

def uvw_from_spherical(theta_deg, phi_deg, flatten=False):
    """ Returns a grid of UV projected points from a list of theta and phi coordinates in degrees.
    """
    theta_smp_rad = np.deg2rad(theta_deg)
    phi_smp_rad = np.deg2rad(phi_deg)

    u = np.sin(theta_smp_rad) * np.cos(phi_smp_rad)
    v = np.sin(theta_smp_rad) * np.sin(phi_smp_rad)
    w = np.cos(theta_smp_rad) * np.ones(u.shape)

    return u, v, w

def spherical_from_uv(u, v):
    """ Converts a u,v projected coordinate to theta, phi in degrees.
        Theta is calculated with the standard cartesian to spherical conversion, but the phi calculation is modified a bit to not require z.
    """
    phi = np.arctan2(v,u)
    # derive this by adding the equations for u^2 and v^2.
    # u^2 + v^2 = sin^2(theta)(sin^2(phi) + cos^2(phi))
    theta = np.arcsin(np.sqrt(u**2 + v**2))

    return np.rad2deg(theta), np.rad2deg(phi)

pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)

sys.argv = sys.argv[0:1]

u0 = const.u0
e0 = const.e0
c0 = const.c0
eta0 = const.eta0

f0 = 10e9
lam0 = const.c0_in / f0

ms_w = 0.02
ms_len = 0.9
ms1_y = 0

sbox_h = lam0
sbox_w = lam0
sbox_len = lam0

gap = 0.01
ms_x = (0, 0)
ms_y = (-ms_w / 2, ms_w / 2)
ms1_z = (gap/2, (lam0 / 4) * 0.9)
ms2_z = (-(lam0 / 4) * 0.9, -gap/2)


sbox = pv.Cube(center=(0, 0, 0), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

ms_upper = pv.Rectangle([
    (ms_x[0], ms_y[0], ms1_z[0]),
    (ms_x[0], ms_y[1], ms1_z[0]),
    (ms_x[1], ms_y[1], ms1_z[1])
])

ms_lower = pv.Rectangle([
    (ms_x[0], ms_y[0], ms2_z[0]),
    (ms_x[0], ms_y[1], ms2_z[0]),
    (ms_x[1], ms_y[1], ms2_z[1])
])


port1_face = pv.Rectangle([
    (ms_x[0], ms_y[0], gap/2),
    (ms_x[0], ms_y[1], gap/2),
    (ms_x[1], ms_y[1], -gap/2)
])


s = rfn.FDTD_Solver(sbox)
# s.add_dielectric("sub", substrate, er=1, style=dict(opacity=0.0))
s.add_conductor(ms_upper, ms_lower, style=dict(color="gold"))
s.add_lumped_port(1, port1_face, "z-")

s.assign_PML_boundaries("x-", "x+", "y-", "y+", "z+", "z-", n_pml=5)

self = s
# having three cells in the PEC instead of 4 causes the edge correction to fail
# s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=4, d_sub=0.01, n_min_sub=4, blend_pec=False)
s.generate_mesh(d0 = 0.02, d_edge=0.01)

s.add_field_monitor("e_tot", "e_total", "y", 0, n_step=10)


ff_bounds = np.array([(-0.35, 0.35), (-0.35, 0.35), (-0.35, 0.35)])
ff_box = pv.Box(
    bounds = ff_bounds.flatten()
)

self.add_farfield_monitor(ff_box, [10e9, 11e9])

plotter = s.render(show_mesh=True, show_probes=True)

plotter.show()


vsrc = 1e-2 * self.gaussian_modulated_source(f0, width=400e-12, t0=200e-12, t_len=800e-12)

# plt.plot(vsrc)
s.assign_excitation(vsrc, 1)
s.solve(n_threads=4)

ff_data = s.get_farfield_data(theta=np.arange(-180, 181), phi=0)

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="polar"))
# plt.plot(np.deg2rad(theta_range), rfn.conv.db20_lin(E_phi))
ax.plot(np.deg2rad(ff_data.coords["theta"]), rfn.conv.db20_lin(ff_data[0, 0]).squeeze())
ax.plot(np.deg2rad(ff_data.coords["theta"]), rfn.conv.db20_lin(ff_data[0, 1]).squeeze())
ax.set_theta_zero_location('N') 
ax.set_theta_direction(-1) 
ax.set_ylim([-40, 10])

ax.set_xlabel("$\\theta$ [deg]")

#%%
cpos = pv.CameraPosition(
    position=(-1.2, -2.5, 0.5),
    focal_point=(0, 0, 0),
    viewup=(0, 0.0, 1.0),
)

gif_setup = dict(file="dipole.gif", fps=10, start_ps=0, end_ps=700, step_ps=5)
p = s.plot_monitor(
    ["e_tot", "e_tot"], 
    opacity=["linear", None], 
    style=["vectors", "surface"], 
    vmin=-30, vmax=20, 
    show_mesh=False, show_rulers=False,
    camera_position=cpos,
    # gif_setup=gif_setup
)
p.add_mesh(ff_box, style="wireframe")
# p.show()


frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)
sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]

fig, ax = plt.subplots()
plt.plot(frequency / 1e9, conv.db20_lin(S11))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S11"])

plt.show()
# %%

