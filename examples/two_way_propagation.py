import pyvista as pv
import vtk
from np_struct import ldarray
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rfnetwork as rfn

from pathlib import Path

from rfnetwork import const, conv, utils, core, utils_mesh, antennas

dir_ = Path(__file__).parent

eta0 = 377


def visible_faces(obj: pv.PolyData, source_pos: tuple) -> np.ndarray:
    """
    Return a boolean mask, shape (n_cells,), that is True for faces whose center has a clear line of
    sight to the source, i.e. the first face hit by a ray from the source towards the face center is the face itself.
    """
    source_pos = np.asarray(source_pos, dtype=float)
    obj_centers = obj.cell_centers().points

    # spatial search structure over the mesh faces, built once and reused for every ray
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(obj)
    locator.BuildLocator()

    # output references filled in by IntersectWithLine, only the hit cell id is used
    t, x, pcoords = vtk.reference(0.0), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    sub_id, cell_id, cell = vtk.reference(0), vtk.reference(0), vtk.vtkGenericCell()

    visible = np.zeros(obj.n_cells, dtype=bool)
    for i, center in enumerate(obj_centers):
        # extend the ray slightly past the face center so the face itself is always intersected
        ray_v = center - source_pos
        ray_end = center + 1e-3 * ray_v / np.linalg.norm(ray_v)
        # IntersectWithLine returns the intersection closest to the start of the ray
        locator.IntersectWithLine(source_pos, ray_end, 1e-9, t, x, pcoords, sub_id, cell_id, cell)
        visible[i] = cell_id.get() == i

    return visible


def polar_dir(theta, phi):
    """Unit vectors in the global frame for each theta of a constant phi cut, u = sin(th)cos(ph), v = sin(th)sin(ph), w = cos(th)"""
    th_r, ph_r = np.deg2rad(theta), np.deg2rad(phi)
    return np.stack([np.sin(th_r) * np.cos(ph_r), np.sin(th_r) * np.sin(ph_r), np.cos(th_r) * np.ones_like(th_r)], axis=-1)


# def add_polar_lines(plotter: pv.Plotter, J_xyz, M_xyz, frequency):

#     theta = np.arange(-180, 182, 2)
#     rE_local_phi0 = antennas.get_farfield_rE(scatterer1, J_xyz, M_xyz, theta, np.ones_like(theta) * 0, frequency).squeeze()
#     rE_local_phi90 = aget_farfield_rE(scatterer1, J_xyz, M_xyz, theta, np.ones_like(theta) * 90, frequency).squeeze()

#     # 2D polar patterns of the scattered field, drawn in the scene around the scatterer.
#     # total field magnitude (thetapol and phipol combined), both cuts normalized to the common peak
#     polar_cuts = {0: rE_local_phi0, 90: rE_local_phi90}
#     polar_mag = {ph: np.sqrt(np.sum(np.abs(rE) ** 2, axis=0)) for ph, rE in polar_cuts.items()}
#     polar_peak = max(np.max(m) for m in polar_mag.values())
#     # map dB values to a radius, floor at polar_range_db below the peak
#     polar_range_db = 30
#     polar_radius = 1.5  # radius in meters of the 0 dB ring
#     polar_grid_db = np.arange(-polar_range_db + 10, 1, 10)  # dB values of the grid rings
#     polar_grid_spokes = np.arange(-180, 180, 30)  # theta values of the grid spokes

#     polar_lines, polar_grids = [], []
#     for ph, mag in polar_mag.items():
#         rE_db = conv.db20_lin(mag / polar_peak)
#         polar_r = polar_radius * np.clip(rE_db + polar_range_db, 0, None) / polar_range_db
#         polar_lines.append(pv.lines_from_points(scatterer1_center + polar_r[:, None] * polar_dir(theta, ph)))

#         # rings at constant dB in the plane of the cut
#         for db in polar_grid_db:
#             ring_r = polar_radius * (db + polar_range_db) / polar_range_db
#             polar_grids.append(pv.lines_from_points(scatterer1_center + ring_r * polar_dir(theta, ph)))
#         # spokes from the center out to the 0 dB ring
#         for th in polar_grid_spokes:
#             polar_grids.append(pv.Line(scatterer1_center, scatterer1_center + polar_radius * polar_dir(th, ph)))

#     # phi=0 cut in blue, phi=90 cut in red, same order as polar_cuts
#     for line, color in zip(polar_lines, ["blue", "red"]):
#         plotter.add_mesh(line, color=color, line_width=2, opacity=0.5)
#     plotter.add_mesh(pv.merge(polar_grids), color="gray", opacity=0.2, line_width=1)

def add_ground_plane(plotter, width, length):
    # --- Create a ground plane ---
    plane = pv.Plane(
        center=(0, 0, length/2),
        direction=(0, 1, 0),
        i_size=length,
        j_size=width,
        i_resolution=1,
        j_resolution=1,
    )

    # Generate planar texture coordinates
    plane.texture_map_to_plane(inplace=True)

    # --- Generate a procedural dirt texture ---
    rng = np.random.default_rng(42)
    size = 256

    # Smooth, mottled noise
    noise = rng.random((size, size))
    for _ in range(4):
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, -1, axis=1)
        ) / 5

    # Add fine-grained variation
    noise = 0.75 * noise + 0.25 * rng.random((size, size))
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    # Convert noise into earthy RGB colors
    dark = np.array([85, 65, 43])
    light = np.array([160, 133, 95])

    texture_image = (
        dark[None, None, :]
        + noise[:, :, None] * (light - dark)[None, None, :]
    ).clip(0, 255).astype(np.uint8)

    texture = pv.Texture(texture_image)
    texture.repeat = True

    plotter.add_mesh(
        plane,
        texture=texture,
        show_edges=False,
        opacity=0.5
    )

##############################
# Build Scene
##############################
frequency = 3e9
lam0 = rfn.const.c0 / frequency
beta = 2 * np.pi / lam0

pv.set_jupyter_backend("trame")

# TX beam pattern, x-polarization
tx_patt = ldarray.load(dir_ / "data/vivaldi_antenna_patt.npy")
tx_pin = 1 # input power in watts

# move the tx pattern to the source position, currently centered at the origin
antenna_height = 0.5
source_pos = (0, antenna_height, 0)
src_patt = rfn.antennas.translate(tx_patt, source_pos)

# add antenna rendering
antenna = pv.Cube(x_length=0.3, y_length=0.4, z_length=0.1)
antenna.points += source_pos

# build scene, dimensions in meters
# z points out from antenna, x is azimuth, y is elevation (antenna centered coordinate frame)
scatterer1 = pv.read(pv.examples.planefile)
# scatterer1 = pv.Rectangle(
#     ((-1, -1, 0), (-1, 1, 0), (1, 1, 0))
# )
# scatterer1 = pv.Sphere(radius=1, theta_resolution=100, phi_resolution=100)

# normalize size to 1m
scatterer1.points /= np.max(np.abs(scatterer1.points))
# scale to desired size
scatterer1.points *= 1
# center at 0
scatterer1.points -= np.average(scatterer1.points, axis=0)

# uniformly subdivide so that no face is larger than max_cell_area. Each linear subdivision splits every
# triangle into 4, quartering the area, so the number of levels is set by the largest face.
max_cell_area = lam0 ** 2
scatterer1 = scatterer1.triangulate()
cell_area_max = np.max(scatterer1.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"])
n_subdiv = int(max(0, np.ceil(np.log(cell_area_max / max_cell_area) / np.log(4))))

# scatterer1 = scatterer1.subdivide(3, subfilter="linear")

# # rotate to desired orientation with center at origin
rot = Rotation.from_euler("xyz", (-90, 70, 20), degrees=True).as_matrix()
scatterer1.points = np.einsum("ij, ...j->...i", rot, scatterer1.points)

# move to desired location
scatterer1.points += (-1, 1.5, 10)


##############################
# Illuminate scene from transmitter
##############################
J_xyz_dir, M_xyz_dir, r = antennas.illuminate_object(scatterer1, source_pos, tx_patt, 1, frequency)
J_xyz_im, M_xyz_im, r_im = antennas.illuminate_object(scatterer1, source_pos, tx_patt, 1, frequency, image=True)

J_xyz = J_xyz_dir + J_xyz_im
M_xyz = M_xyz_dir + M_xyz_im


scatterer1_center = np.average(scatterer1.cell_centers().points, axis=0)

plotter = pv.Plotter()
scatterer1.cell_data["Js"] = rfn.conv.db20_lin(np.linalg.norm(J_xyz, axis=-1)) #np.abs(J_xyz[..., 1])
scatterer1.cell_data["Ms"] = rfn.conv.db20_lin(np.linalg.norm(M_xyz, axis=-1))

# ground plane
# gnd = pv.Rectangle(((-2.5, 0, 0), (2.5, 0, 0), (2.5, 0, 12)))
add_ground_plane(plotter, 5, 12)

plotter.add_mesh(antenna, color="tan")
# plotter.add_mesh(gnd, color="k", opacity=0.2)
plotter.add_mesh(scatterer1, cmap="jet", scalars="Ms", clim=(-30, 20))
# add_polar_lines(plotter, J_xyz, M_xyz, frequency)
plotter.add_axes()
# plotter.show_bounds(bounds=(-2.5, 2.5, 0, 3.5, 0, 12), show_zlabels=False, show_xlabels=False, show_ylabels=False, xtitle=" ", ytitle=" ", ztitle=" ")
# plotter.show_grid()

# plotter.add_bounding_box()


cpos = pv.CameraPosition(position=(-7, 6, 6), focal_point=(0.3379, 0.8740, 10), viewup=(0, 1, 0))
plotter.camera_position = cpos
plotter.show()

# fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
# rE_local_phi0.plot("theta", xfmt=np.deg2rad, yfmt="db20")
# plt.show()

##############################
# Propagate back to RX antennas
##############################
# rx antenna pattern
rx_pos = [
    (-0.1, antenna_height - 0.1, 0), 
    (-0.1, antenna_height + 0.1, 0),
    (0.1, antenna_height - 0.1, 0),
    (0.1, antenna_height + 0.1, 0)
]

rx_pos = ldarray(rx_pos, coords=dict(ele=range(4), axis=("x", "y", "z")))
rx_patt = rfn.antennas.translate(tx_patt, rx_pos)

# vector from center of scatterer to rx antenna
obj_rx_p = rx_pos - scatterer1_center
# distance from scatterer to antenna
obj_rx_r = np.linalg.norm(obj_rx_p, axis="axis")
# normalized uvw position of rx antenna from the local coordinate frame of scatterer
obj_rx_v = obj_rx_p / obj_rx_r

# theta and phi angle back to the rx antenna from scatterer
phi_rx, theta_rx = rfn.antennas.phitheta_uvw(*(obj_rx_v.T), flat=True)

rE_at_rx, rx_r = antennas.get_scattering_rxE(scatterer1, J_xyz, M_xyz, rx_patt, rx_pos, frequency)
# convert to rectangular polarization
rE_at_rx = rfn.antennas.pattern_spherical2rectangular(rE_at_rx).squeeze().rename(phitheta="ele")

# propagate rE to get e-field at source
e_inc_rx = rE_at_rx * np.exp(1j * beta * obj_rx_r) / obj_rx_r

# get incident power E x H = E x (E / eta)
# s_scattered = e_scattered * np.conjugate(e_scattered) / eta0

# gain of rx pattern at incident direction from scatterer
rx_az, rx_el = rfn.antennas.azel_uvw(*(-obj_rx_v.T), flat=True)
rx_patt_inc = rx_patt.sel(frequency=frequency).interpolate(az=rx_az, el=rx_el, ele=np.arange(4), flat=True).squeeze()
# effective voltage area of rx antenna
# e_eff = G (lam^2 / 4pi)
eff_area_rx = np.abs(rx_patt_inc).rename(azelele="ele") * np.sqrt(lam0**2 / (4 * np.pi))
# received field at antenna terminals.
# divide by eta0 so when the result is squared it's equal to the power
# at the terminals, E x H = E x E* / eta0
e_rx_terminal = eff_area_rx * e_inc_rx / np.sqrt(eta0)

################
# solve radar equation for radar cross section
################
#  {P_{r}={{P_{t}G_{t}} \over {4\pi r^{2}}}\sigma {{1} \over {4\pi r^{2}}}A_{\mathrm {eff} }}

# power received
pwr_rx = np.abs(e_rx_terminal)**2

# tx antenna gain pointing towards object
tx_obj_p = scatterer1_center - source_pos
tx_obj_r = np.linalg.norm(tx_obj_p)
tx_obj_v = tx_obj_p / tx_obj_r
tx_az, tx_el = rfn.antennas.azel_uvw(*tx_obj_v, flat=True)
tx_gain_v = tx_patt.interpolate(az=tx_az, el=tx_el).sel(frequency=frequency).squeeze()

# radar range equation for each polarization, first rx antenna
sig_rcs = (pwr_rx / (tx_pin * np.abs(tx_gain_v)**2 * eff_area_rx)) * ((4 * np.pi * tx_obj_r**2)**2) 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8))
e_rx_terminal.plot("ele", yfmt="db20", marker="o", ax=ax1)
e_rx_terminal.plot("ele", yfmt="deg", marker="o", ax=ax2)
sig_rcs.plot("ele", yfmt="db10", polarization="y", marker="o", ax=ax3, linestyle="none")
ax1.set_title("|Er|")
ax2.set_title("|Er|")
ax3.set_title("rcs, y-pol")
ax1.set_ylim([-150, -50])
ax3.set_ylim([-50, 0])
ax3.set_ylabel("dBsm")
fig.tight_layout()
plt.show()