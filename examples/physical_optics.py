import pyvista as pv
import vtk
from np_struct import ldarray
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rfnetwork as rfn

from pathlib import Path

from rfnetwork import const, conv, utils, core, utils_mesh

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

def illuminate_object(
        obj: pv.PolyData, source_pos: tuple, src_patt: ldarray, source_pwr: float, frequency: float, image: bool = False, ray_trace: bool = False
    ):

    # flip location of source to the other side of ground plane if computing the image
    if image:
        source_pos = list(source_pos)
        source_pos[1] = -source_pos[1]
        src_patt = -src_patt

    # center of each scatterer face, shape (n_cells, 3)
    obj_centers = obj.cell_centers().points
    # unit normal of each scatterer face, pointing out of the surface, shape (n_cells, 3)
    obj_face_normals = obj.cell_normals
    # vector from source location to scatterer
    obj_src_p = obj_centers - source_pos
    # distance r from source to center of each face
    obj_src_r = np.linalg.norm(obj_src_p, axis=-1)
    # normalized vectors from source to each face (direction only)
    poynting_v = (obj_src_p) / obj_src_r[..., None]

    # azimuth angle of each face center from the z axis (antenna boresight), positive towards +x, in degrees
    obj_az = np.rad2deg(np.arctan2(obj_src_p[:, 0], obj_src_p[:, 2]))
    # elevation angle of each face center above the xz plane, positive towards +y, in degrees.
    # matches the rfnetwork az/el convention: u = cos(el) sin(az), v = sin(el), w = cos(el) cos(az)
    obj_el = np.rad2deg(np.arctan2(obj_src_p[:, 1], obj_src_p[:, 2]))

    # flip elevation if image
    if image:
        obj_el = -obj_el

    # evaluate tx pattern at scatterer faces, remove frequency axis
    gain_at_faces = src_patt.sel(frequency=frequency).interpolate(az=obj_az, el=obj_el, flat=True)

    # the gain is related to the e-field strength by
    # G = (4pi / 2 eta0) (|rE|^2 / Pin)
    # E = sqrt(Pin G (2 eta0 / 4pi r^2))
    # the tx_patt is already complex voltage gain (sqrt(G))
    # E = g * sqrt(Pin (2 eta0 / 4pi r^2))
    e_inc = gain_at_faces * np.sqrt(source_pwr * (2 * eta0 / (4 * np.pi * obj_src_r**2))) * np.exp(-1j * beta * obj_src_r)
    e_inc = e_inc.transpose(("azel", "polarization"))
    # cross product with propagation direction vector to get incident h field
    # TODO: I think this should be k X E, not -k X E. This is probably correcting a sign flip somewhere downstream
    h_inc = np.cross(-poynting_v, e_inc) / eta0

    # compute intercepted field using the dot product of face normal with position vector
    eff_area = np.einsum("ij,ij->i", obj_face_normals, poynting_v)
    # clip values that are positive. If face normal is in same direction as the position vector,
    # it's dot product is positive
    eff_area = np.abs(np.where(eff_area > 0 , 0, eff_area))

    # compute surface currents on each face
    # physical optics electric surface current on the PEC scatterer, Js = 2 n^ x H_inc on lit faces
    js_at_faces = np.cross(obj_face_normals, h_inc) * eff_area[..., None]
    ms_at_faces = np.cross(-obj_face_normals, e_inc) * eff_area[..., None]

    # A face is lit if it points towards the source
    # and no other part of the object blocks the line of sight to it.
    if ray_trace:
        lit = visible_faces(obj, source_pos)
        js_at_faces *= lit[..., None]
        ms_at_faces *= lit[..., None]

    return js_at_faces, ms_at_faces



def get_farfield_rE(
        J_xyz, M_xyz, theta: np.ndarray, phi: np.ndarray, frequency: np.ndarray = None, n_threads: int = 4) -> ldarray:
    """
    Compile E-field monitor data from the farfield monitor attached to the solver.

    E-field values are returned for thetapol and phipol, and are normalized by (exp(1j * beta * r) / r).

    Parameters
    ----------
    theta : np.ndarray | float
        spatial theta values in degrees

    phi : np.ndarray | float
        spatial phi values in degrees

    Returns
    -------
    ldarray
        rE field values. labeled numpy array with dimensions (polarization, frequency, theta, phi)
    """

    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    frequency = np.atleast_1d(frequency)

    cell_pos = np.array(obj.cell_centers().points.T, order="C", dtype=np.float32)

    # area of each face
    sized = obj.compute_cell_sizes(length=False, area=True, volume=False)
    cell_ds = np.array(sized.cell_data["Area"], order="C", dtype=np.float32)   # n_cells

    # max grid length for temporary working grid
    n_cells = obj.n_cells

    ff_data = dict(
        beta = np.array(2 * np.pi * frequency / rfn.const.c0, dtype=np.float32, order="C"),
        theta = np.array(np.deg2rad(theta), dtype=np.float32, order="C"),
        phi = np.array(np.deg2rad(phi), dtype=np.float32, order="C"),
        data = np.zeros((2, 1, len(theta), len(phi)), dtype=np.complex64, order="C"),
        working_grid_cmplx = np.zeros((n_threads, n_cells), dtype=np.complex64, order="C"),
        working_grid_float = np.zeros((n_threads, n_cells), dtype=np.float32, order="C")
    )

    J_xyz_t = np.array(J_xyz.T, order="C", dtype=np.complex64)
    M_xyz_t = np.array(M_xyz.T, order="C", dtype=np.complex64)

    core.core_func.nf2ff_triangulated(J_xyz_t, M_xyz_t, cell_pos, cell_ds, ff_data, n_threads)

    # cast as labeled array
    return ldarray(
        ff_data["data"],
        coords=dict(polarization=["thetapol", "phipol"], frequency=frequency, theta=theta, phi=phi)
    )


f0 = 3e9
lam0 = rfn.const.c0 / f0
beta = 2 * np.pi / lam0

pv.set_jupyter_backend("trame")

# TX beam pattern, x-polarization
tx_patt = ldarray.load(dir_ / "data/vivaldi_antenna_patt.npy")
tx_pin = 1 # input power in watts

# build scene, dimensions in meters
# z points out from antenna, x is azimuth, y is elevation (antenna centered coordinate frame)
# scatterer1 = pv.read(pv.examples.planefile)
# scatterer1 = pv.Rectangle(
#     ((-1, -1, 0), (-1, 1, 0), (1, 1, 0))
# )
# # subdivide so that no face edge is longer than lambda/6, keeps the phase variation across each face small.
# # the scattered field at the antenna converges to within 0.1 dB and 0.5 deg at this density.
# scatterer1 = scatterer1.triangulate().subdivide(7)#.subdivide_adaptive(max_edge_len=lam0 / 4, max_tri_area=(lam0 / 4) ** 2)


scatterer1 = pv.Sphere(radius=1, theta_resolution=100, phi_resolution=100)

# normalize size to 1m
scatterer1.points /= np.max(np.abs(scatterer1.points))
# scale to desired size
scatterer1.points *= 0.6
# center at 0
scatterer1.points -= [np.average(scatterer1.points[:, i]) for i in range(3)]

# # rotate to desired orientation with center at origin
rot = Rotation.from_euler("x", -30, degrees=True).as_matrix()
scatterer1.points = np.einsum("ij, ...j->...i", rot, scatterer1.points)

# move to desired location
scatterer1.points += (0, 0, 1)

# move the tx pattern to the source position, currently centered at the origin
antenna_height = 0
source_pos = (0, antenna_height, 0)
src_patt = rfn.antennas.translate(tx_patt, source_pos)

# add antenna rendering
antenna = pv.Cube(x_length=0.3, y_length=0.3, z_length=0.1)
antenna.points += source_pos

obj = scatterer1
source_pwr = 1
image = False
frequency=f0
ray_trace = False



J_xyz, M_xyz = illuminate_object(scatterer1, source_pos, tx_patt, 1, frequency)

plotter = pv.Plotter()
scatterer1.cell_data["Ms (dB)"] = np.abs(M_xyz[..., 0])
scatterer1.cell_data["Ms (ang)"] = np.angle(M_xyz[..., 1], deg=True)

plotter.add_mesh(antenna)
plotter.add_mesh(scatterer1, cmap="jet", scalars="Ms (dB)")
plotter.add_axes()
plotter.camera_position = "iso"
plotter.show()


obj = scatterer1
theta = np.arange(-180, 182, 2)
phi = 90
n_threads = 4

rE = get_farfield_rE(J_xyz, M_xyz, theta, phi, frequency)

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

rE.plot("theta", xfmt=np.deg2rad, yfmt="db20")
plt.show()

