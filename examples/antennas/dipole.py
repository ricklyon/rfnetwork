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

self.add_farfield_monitor(ff_box, 10e9)

plotter = s.render(show_mesh=True, show_probes=True)

plotter.show()







vsrc = 1e-2 * self.gaussian_modulated_source(f0, width=400e-12, t0=200e-12, t_len=800e-12)

plt.plot(vsrc)
s.assign_excitation(vsrc, 1)
s.solve(n_threads=4)

s.get_farfield_data(0, 0)


# axis = 0
# j = 0 
# f = 1
# side = "n"
# monitors = [m for (k, m) in self.monitors.items() if k.startswith("ff_")]
# # indices of each face on farfield box
# ff_idx = self.farfield["idx"]

# def get_farfield_data(self, theta, phi):

#     # equivalent currents at the cell centers, two faces per axis
#     J_xyz = [[None, None] for i in range(3)]
#     M_xyz = [[None, None] for i in range(3)]

#     # surface position, two faces per axis, meters
#     surf_pos = [[None, None] for i in range(3)]

#     # meshgrid of x/y positions on grid, same for each face on the same axis
#     r_grid = [[None, None] for i in range(3)]

#     # meshgrid of cell widths along each surface axis, same for each face on the same axis
#     w_grid = [[None, None] for i in range(3)]

#     # initialize matrix for far-field data
#     frequency = self.farfield["frequency"]
#     n_frequencies = len(frequency)
#     ff_data = np.zeros((2, n_frequencies, len(theta), len(phi)), dtype=np.complex128, order="C")

#     for axis in range(3):

#         # field components on surface
#         axis_s = ("x", "y", "z")[axis]
#         sf0, sf1 = [i for i in (0, 1, 2) if i != axis]
#         sf0_s, sf1_s = [a for a in ("x", "y", "z") if a != axis_s]

#         # grid cell positions
#         r_grid[axis][0] = np.array(self.farfield["cell_pos"][axis][0], dtype=np.float32, order="C")
#         r_grid[axis][1] = np.array(self.farfield["cell_pos"][axis][1], dtype=np.float32, order="C")

#         # grid cell widths
#         w_grid[axis][0] = np.array(self.farfield["cell_w"][axis][0], order="C")
#         w_grid[axis][1] = np.array(self.farfield["cell_w"][axis][1], order="C")

#         # for each face on either side of the far-field box
#         for j, side in enumerate(["n", "p"]):
#             # surface position, meters
#             surf_pos[axis][j] = self.farfield["surf_pos"][axis, j]

#             # shape of the grid cells on surface
#             surf_shape = self.farfield["cell_pos"][axis][0].shape
            
#             # initialize surface field at the cell centers
#             e_xyz = np.zeros(((3, n_frequencies) + surf_shape), dtype=np.complex128, order="C")
#             h_xyz = np.zeros(((3, n_frequencies) + surf_shape), dtype=np.complex128, order="C")
            
#             for f in (sf0, sf1):
#                 # string value for field direction
#                 f_s = ("x", "y", "z")[f]

#                 # near-field monitor names
#                 emon = f"ff_e{f_s}_{side}{axis_s}"
#                 hmon1 = f"ff_h{f_s}1_{side}{axis_s}"
#                 hmon2 = f"ff_h{f_s}2_{side}{axis_s}"

#                 # skip faces that are on solve box boundaries
#                 if emon not in s.monitors.keys():
#                     continue
                
#                 # get near-field data
#                 edata = s.get_monitor_data(emon)
#                 hdata1 = s.get_monitor_data(hmon1)
#                 hdata2 = s.get_monitor_data(hmon2)

#                 # widths of the cells that the h-components are in, along the axis
#                 hidx1 = self.monitors[hmon1]["index"]
#                 hidx2 = self.monitors[hmon2]["index"]
#                 w1, w2 = self.d_cells[axis][hidx1], self.d_cells[axis][hidx2]
#                 # average the two h field monitor surfaces to get the values at the same location as 
#                 # the e-fields. 
#                 hdata = (hdata1 * (w1/2) + hdata2 * (w2/2)) / ((w1/2) + (w2/2))

#                 # average e field along opposite surface axis to get the fields on the cell center
#                 left_idx, right_idx = [slice(None), slice(None), slice(None)], [slice(None), slice(None), slice(None)]
#                 avg_axis = int(1 if f == sf1 else 2)
#                 left_idx[avg_axis] = slice(1, None)
#                 right_idx[avg_axis] = slice(None, -1)

#                 edata_cell = (edata[tuple(left_idx)] + edata[tuple(right_idx)]) / 2
#                 # get values inside the solve box
#                 e_xyz[f] = edata_cell[:, ff_idx[sf0, 0]: ff_idx[sf0, 1], ff_idx[sf1, 0]: ff_idx[sf1, 1]]

#                 # average h field along the field axis to get field at the cell center
#                 left_idx, right_idx = [slice(None), slice(None), slice(None)], [slice(None), slice(None), slice(None)]
#                 avg_axis = int(1 if f == sf0 else 2)
#                 left_idx[avg_axis] = slice(1, None)
#                 right_idx[avg_axis] = slice(None, -1)

#                 hdata_cell = (hdata[tuple(left_idx)] + hdata[tuple(right_idx)]) / 2
#                 # get values inside the solve box
#                 h_xyz[f] = hdata_cell[:, ff_idx[sf0, 0]: ff_idx[sf0, 1], ff_idx[sf1, 0]: ff_idx[sf1, 1]]

#             # the normal axis vector, points out from surface
#             normal_axis_v = np.array([0, 0, 0])
#             normal_axis_v[axis] = (-1 if j == 0 else 1)
#             # magnetic equivalent surface currents, n X Hs
#             # equation 7-43 in Balanis Fields and Waves
#             J_xyz[axis][j] = np.cross(normal_axis_v, h_xyz, axis=0)
#             # electric equivalent surface currents, -n X Es
#             M_xyz[axis][j] = np.cross(-normal_axis_v, e_xyz, axis=0)


#     beta = np.array(2 * np.pi * frequency / rfn.const.c0, dtype=np.float32)
#     theta = np.array(np.deg2rad(theta), dtype=np.float32)
#     phi = np.array(np.deg2rad(phi), dtype=np.float32)

#     # core.core_func.nf2ff(J_xyz, M_xyz, r_grid, w_grid, surf_pos, beta, theta, phi, ff_data)




# #%%
# # Calculate far-field terms
# theta_range = np.arange(-180, 181, 1)

# E_theta = np.zeros(len(theta_range), dtype="complex128")
# E_phi = np.zeros(len(theta_range), dtype="complex128")

# frequency = self.farfield["frequency"]
# phi_deg = 90
# beta = 2 * np.pi * frequency / rfn.const.c0

# theta_deg = 0
# i = 0
# j = 0
# axis = 0
# side = "n"

# for i, theta_deg in enumerate(theta_range):

#     th = np.deg2rad(theta_deg)
#     ph = np.deg2rad(phi_deg)

#     cos_th = np.cos(th)
#     sin_th = np.sin(th)
#     cos_ph = np.cos(ph)
#     sin_ph = np.sin(ph)

#     u, v, w = uvw_from_spherical(theta_deg, phi_deg)

#     N_theta = 0
#     N_phi = 0
#     L_theta = 0 
#     L_phi = 0

#     ff_idx = self.farfield["idx"]

#     for axis in range(3):

#         # field components on surface
#         axis_s = ("x", "y", "z")[axis]
#         sf0, sf1 = [i for i in (0, 1, 2) if i != axis]
#         sf0_s, sf1_s = [a for a in ("x", "y", "z") if a != axis_s]
        
#         # for each face on either side of the far-field box
#         for j, side in enumerate(["n", "p"]):

#             # r vector of grid cell positions
#             r_pos = [None] * 3 
#             r_pos[axis] = self.farfield["surf_pos"][axis, j]
#             r_pos[sf0] = self.farfield["cell_pos"][axis][0]
#             r_pos[sf1] = self.farfield["cell_pos"][axis][1]
#             # phase term for integrand
#             phs_term = np.exp(1j * beta.item() * (r_pos[0] * u + r_pos[1] * v + r_pos[2] * w))
            
#             # surface field at the cell centers
#             e_xyz = np.zeros(((3,) + phs_term.shape), dtype=np.complex128)
#             h_xyz = np.zeros(((3,) + phs_term.shape), dtype=np.complex128)
#             for f in (sf0, sf1):
#                 # string value for field direction
#                 f_s = ("x", "y", "z")[f]

#                 edata = s.get_monitor_data(f"ff_e{f_s}_{side}{axis_s}")
#                 # average the two h field monitor surfaces to get the values at the same location as 
#                 # the e-fields. Assumes cells are equal width in the free space region around the ff box
#                 hdata1 = s.get_monitor_data(f"ff_h{f_s}1_{side}{axis_s}")
#                 hdata2 = s.get_monitor_data(f"ff_h{f_s}1_{side}{axis_s}")

#                 hdata = (hdata1 + hdata2) / 2

#                 # average e field along opposite surface axis to get the fields on the cell center
#                 left_idx, right_idx = [slice(None), slice(None), slice(None)], [slice(None), slice(None), slice(None)]
#                 avg_axis = int(1 if f == sf1 else 2)
#                 left_idx[avg_axis] = slice(1, None)
#                 right_idx[avg_axis] = slice(None, -1)

#                 edata_cell = (edata[tuple(left_idx)] + edata[tuple(right_idx)]) / 2
#                 # get values inside the solve box
#                 e_xyz[f] = edata_cell[:, ff_idx[sf0, 0]: ff_idx[sf0, 1], ff_idx[sf1, 0]: ff_idx[sf1, 1]]

#                 # average h field along the field axis to get field at the cell center
#                 left_idx, right_idx = [slice(None), slice(None), slice(None)], [slice(None), slice(None), slice(None)]
#                 avg_axis = int(1 if f == sf0 else 2)
#                 left_idx[avg_axis] = slice(1, None)
#                 right_idx[avg_axis] = slice(None, -1)

#                 hdata_cell = (hdata[tuple(left_idx)] + hdata[tuple(right_idx)]) / 2
#                 # get values inside the solve box
#                 h_xyz[f] = hdata_cell[:, ff_idx[sf0, 0]: ff_idx[sf0, 1], ff_idx[sf1, 0]: ff_idx[sf1, 1]]

#             # the normal axis vector, points out from surface
#             normal_axis_v = np.array([0, 0, 0])
#             normal_axis_v[axis] = (-1 if j == 0 else 1)
#             # magnetic equvilent surface currents, n X Hs
#             # equation 7-43 in Balanis Fiels and Waves
#             h_xyz[axis] = np.zeros_like(h_xyz[f])
#             Jx, Jy, Jz =  np.cross(normal_axis_v, h_xyz, axis=0)
#             # electric equvilent surface currents, -n X Es
#             Mx, My, Mz =  np.cross(-normal_axis_v, e_xyz, axis=0)

#             # cell widths across the surface
#             d1, d2 = self.farfield["cell_w"][axis][0], self.farfield["cell_w"][axis][1]
#             # ds * e^(jB r cos (psi))
#             dS = phs_term * d1 * d2

#             # N_theta
#             N_theta_intg = (Jx * cos_th * cos_ph) + (Jy * cos_th * sin_ph) - (Jz * sin_th)
#             N_theta += np.sum(N_theta_intg * dS)
#             # N_phi
#             N_phi_intg = (-Jx * sin_ph) + (Jy * cos_ph)
#             N_phi += np.sum(N_phi_intg * dS)
#             # L_theta 
#             L_theta_intg = (Mx * cos_th * cos_ph) + (My * cos_th * sin_ph) - (Mz * sin_th)
#             L_theta += np.sum(L_theta_intg * dS)
#             # L_phi
#             L_phi_intg = (-Mx * sin_ph) + (My * cos_ph)
#             L_phi += np.sum(L_phi_intg * dS)


#     # contributions to N and L have been summed from all faces by this point. Calculate far-field
#     # electric field for thetapol and phipol
#     E_theta[i] = (-j * beta / (4 * np.pi)) * (L_phi + eta0 * N_theta)
#     E_phi[i] = (j * beta / (4 * np.pi)) * (L_theta - eta0 * N_phi)




# fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="polar"))
# # plt.plot(np.deg2rad(theta_range), rfn.conv.db20_lin(E_phi))
# ax.plot(np.deg2rad(theta_range), rfn.conv.db20_lin(E_theta))
# ax.set_theta_zero_location('N') 
# ax.set_theta_direction(-1) 
# ax.set_ylim([-40, 10])

# ax.set_xlabel("$\\theta$ [deg]")

# #%%
# cpos = pv.CameraPosition(
#     position=(-1.2, -2.5, 0.5),
#     focal_point=(0, 0, 0),
#     viewup=(0, 0.0, 1.0),
# )

# gif_setup = dict(file="dipole.gif", fps=10, start_ps=0, end_ps=700, step_ps=5)
# p = s.plot_monitor(
#     ["e_tot", "e_tot"], 
#     opacity=["linear", None], 
#     style=["vectors", "surface"], 
#     vmin=-30, vmax=20, 
#     show_mesh=False, show_rulers=False,
#     camera_position=cpos,
#     # gif_setup=gif_setup
# )
# p.add_mesh(ff_box, style="wireframe")
# # p.show()


# frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)
# sdata = s.get_sparameters(frequency)
# S11 = sdata[:, 0]

# fig, ax = plt.subplots()
# plt.plot(frequency / 1e9, conv.db20_lin(S11))
# mplm.line_marker(x = 10, axes=ax)
# ax.set_xlabel("Frequency [GHz]")
# ax.set_ylabel("[dB]")
# ax.legend(["S11"])

# plt.show()
# # %%
