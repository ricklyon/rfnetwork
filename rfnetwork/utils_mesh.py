
from pathlib import Path
import io
import numpy as np
from np_struct import ldarray
import mpl_markers as mplm
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pyvista as pv
import itertools
import vtk

from PIL import Image

import pygerber.gerberx3.api.v2 as pygb

from . import conv
from .core import core_func

def blend_cell_widths(
    a: float, b: float, d: float, n_min: int = 1, tol: float = 0.0001, dtype_=np.float32
):
    """
    Return a list of cell widths that divide a cell with width d into cells that do not exceed the given
    growth rate r_max. a is the width of the previous cell, b is the width of the next cell.
    """
    cells = core_func.blend_cell_widths(float(a), float(b), float(d), int(n_min), float(tol))

    return cells.astype(dtype_)
    

def get_object_edges(obj: pv.PolyData, group_faces: bool = True, zero_threshold: float = 1e-12) -> list:
    """
    Get the endpoints of the lines that form the edges of the object faces. Returns a M-length list for each 
    face in the object, each containing a N-length list of 2x3 arrays. N is the number of edges in the face.
    Each row is the coordinates of the two endpoints of the edge.

    Parameters
    ----------
    obj : py.PolyData
        pyvista PolyData object 
    group_faces : bool, default: True
        if False, edges from all faces are combined and the returned list is Nx2x3, where N is the 
        number of edges in the object.
    zero_threshold : bool, default: 1e-12
        remove faces with area below this amount. Set to None to return all faces.
    """
    
    # build list of edge coordinates along each axis
    n_cells = np.clip(obj.n_cells, 1, None)
    faces = [list() for n in range(n_cells)]

    # if obj has no faces and is a collection of points, build edges from consecutive points
    if not len(obj.faces):
        obj_edges = []
        for i, p in enumerate(obj.points):
            end_p = obj.points[i+1] if i < len(obj.points) - 1 else obj.points[0]
            obj_edges.append((p, end_p))

        return obj_edges

    else:
        # index of the current face
        face_idx = -1
        # decremented counter, starts at the number of points in the face, reaches zero after the last point
        # in the face and a new face begins.
        face_point_count = 0
        # index of first point in a face
        anchor = None
        # iterate through the list of faces points
        for i, p in enumerate(obj.faces):
            # start a new face
            if face_point_count == 0:
                face_point_count = p
                anchor =  obj.faces[i+1]
                
                if face_idx >= 0:
                    faces[face_idx] += obj_edges
                
                face_idx += 1
                obj_edges = []
            # if on the last point in the face, connect back to the first point in the face
            elif face_point_count == 1:
                obj_edges.append((obj.points[p], obj.points[anchor]))
                face_point_count -= 1
            # connect two points in the face
            else:
                obj_edges.append((obj.points[p], obj.points[obj.faces[i+1]]))
                face_point_count -= 1

        # finish last face
        if len(obj_edges):
            faces[face_idx] += obj_edges

        # prune faces with zero area
        if zero_threshold is not None:
            mesh = obj.compute_cell_sizes()
            face_area = mesh.cell_data["Area"]
            faces = [f for i, f in enumerate(faces) if face_area[i] > zero_threshold]

        # flatten to a list of edges, list of (2x3) matrices
        if not group_faces:
            faces = list(itertools.chain.from_iterable(faces))

    return faces

def is_object_surface(obj: pv.PolyData):
    """
    Returns True if the object is a 2D surface, False if 3D or 1D.
    """

    x0, x1, y0, y1, z0, z1 = obj.bounds
    p0 = x0, y0, z0
    p1 = x1, y1, z1

    axis_len = np.abs(np.diff([p0, p1], axis=0))

    return np.count_nonzero(axis_len > 1e-12) == 2

def remove_interior_edges(obj_edges: np.ndarray, tolerance: float = 1e-6, zero: float = 1e-12):
    """
    Remove interior edges returned from get_object_edges(..., group_faces=False) that share an edge with other faces. 
    Only works for 2D surfaces.
    """
    obj_edges = np.array(obj_edges)
    edges = np.ascontiguousarray(obj_edges.reshape(-1, 2, 3), dtype=np.float64)

    nonovl_edges = core_func.remove_interior_edges(edges, float(tolerance), float(zero))

    return nonovl_edges.astype(obj_edges.dtype)


def get_object_vertices(obj: pv.PolyData, group_faces: bool = True, zero_threshold: float = 1e-12) -> list:
    """
    Get the vertices of each face of the object. Returns a M-length list for each face in the
    object, each containing a Nx3 array of the vertices coordinates in that face.

    Parameters
    ----------
    obj : py.PolyData
        pyvista PolyData object 
    group_faces: bool, default: True
        if False, vertices from all faces are combined and the returned list is Nx2x3, where N is the 
        number of edges in the object.
    zero_threshold : bool, default: True
        remove faces with area below this amount. Set to None to return all faces.
    """
    # build list of edge coordinates along each axis
    n_cells = np.clip(obj.n_cells, 1, None)
    faces = [list() for n in range(n_cells)]

    # if obj has no faces and is a collection of points, build edges from consecutive points
    if not len(obj.faces):
        faces[0] = np.array(obj.points)

    else:
        # index in the list of face points
        face_point_idx = 0
        # index of the face groups
        face_idx = 0
        for i in range(n_cells):
            # number of points in this face
            n_points = obj.faces[face_point_idx]
            
            for p in range(1, n_points + 1):
                faces[face_idx].append(obj.points[obj.faces[face_point_idx + p]])
            
            face_point_idx += (n_points + 1)
            face_idx += 1

        # prune faces with zero area
        if zero_threshold is not None:
            mesh = obj.compute_cell_sizes()
            face_area = mesh.cell_data["Area"]
            faces = [f for i, f in enumerate(faces) if face_area[i] > zero_threshold]

        # flatten to a list of edges, list of (2x3) matrices
        if not group_faces:
            faces = list(itertools.chain.from_iterable(faces))

    return faces


def is_point_in_surface(
    points: np.ndarray, obj: pv.PolyData, tolerance: float = 0.1, d_min: float = None
) -> np.ndarray:
    """
    Returns an array the same shape as points with each value set to True if point is inside the 2D surface.

    Parameters
    ----------
    points : np.ndarray
        matrix of points with the last axis being the xyz position
    obj : py.PolyData
        2D surface to evaluate inclusion for.
    tolerance : float, default: 1
        tolerance in degrees that defines how far outside a point may be from a vertex angle before it is
        not considered part of the surface. Lower values will tend to drop points close to the edge. 
        Value is in degrees.
    d_min : float, optional
        Minimum cell width. If provided, correct points within d_min / 3  of a vertex are included in the surface. 
        Points very close to vertices can be prone to numerical errors in the algorithm. 

    """
    # return all zeros if object mesh is empty
    if not len(obj.points):
        return np.zeros(points.shape[:-1])
    
    # tolerance for zero values
    _zero_tol = 1e-8

    # convert tolerance to radians
    tolerance = np.deg2rad(tolerance)

    points = np.atleast_2d(points)

    # object bounding box
    p0 = np.min(obj.points, axis=0)
    p1 = np.max(obj.points, axis=0)

    # get axis normal to the surface
    normal_axis = (np.diff([p0, p1], axis=0) == 0)
    axis = np.argmax(normal_axis)

    if not np.any(np.abs(normal_axis) > _zero_tol):
        raise ValueError("Object must be a surface on a cardinal plane.")
    
    # get the two axis of the component vectors on the surface
    c1_axis, c2_axis = ((1, 2), (0, 2), (0, 1))[axis]
    
    vertices = get_object_vertices(obj, group_faces=True, zero_threshold=_zero_tol)
    edges = get_object_edges(obj, group_faces=True, zero_threshold=_zero_tol)
    n_faces = len(edges)
    
    # flatten the vertices and edges of each face into contiguous arrays, with the number of vertices and edges in 
    # each face.
    face_vertices = [np.array(vertices[f], dtype=np.float64).reshape(-1, 3) for f in range(n_faces)]
    face_edges = [np.array(edges[f], dtype=np.float64).reshape(-1, 2, 3) for f in range(n_faces)]

    vertex_count = np.array([len(v) for v in face_vertices], dtype=np.int32)
    edge_count = np.array([len(e) for e in face_edges], dtype=np.int32)

    vertices_flat = np.ascontiguousarray(np.concatenate([np.zeros((0, 3))] + face_vertices))
    edges_flat = np.ascontiguousarray(np.concatenate([np.zeros((0, 2, 3))] + face_edges))
    points_flat = np.ascontiguousarray(points.reshape(-1, 3), dtype=np.float64)

    in_surface = core_func.is_point_in_surface(
        points_flat,
        vertices_flat,
        vertex_count,
        edges_flat,
        edge_count,
        int(axis),
        int(c1_axis),
        int(c2_axis),
        float(obj.points[0, axis]),
        float(tolerance),
        -1.0 if d_min is None else float(d_min),
        _zero_tol,
    )

    # return True if point is contained in any face of the surface
    return in_surface.reshape(points.shape[:-1])


def get_gerber_image(filepath: Path, origin: tuple = None, dpi: int = 1000) -> ldarray:
    """
    Get the an image of a single layer gerber file. Pixels in a copper region are set to 1 in the returned array,
    and are set to 0 outside copper regions.

    Parameters
    ----------
    filepath : Path
        path to gerber file
    origin : tuple, default: (0, 0)
        physical x/y location on mesh grid to place the lower left corner of the file. 
    dpi : int, default: 1000
        pixels per inch of rasterized gerber image.

    Returns
    -------
    ldarray :
        labeled numpy array of gerber image. Value of 1 indicates metalized area, 0 indicates etched area.
    """
    # convert dpi to dots per mm
    dpmm = int(dpi * conv.in_mm(1))
    # render gerber as raster image
    gerber = pygb.GerberFile.from_file(filepath).parse()

    buff = io.BytesIO()
    gerber.render_raster(
        buff, image_format=pygb.ImageFormatEnum.PNG, color_scheme=pygb.ColorScheme.COPPER, dpmm=dpmm
    )
    img_raw = np.array(Image.open(buff))

    # copper region color in the raster image
    gcolor = np.array(pygb.DEFAULT_COLOR_MAP[pygb.FileTypeEnum.COPPER].solid_region_color.as_rgb_int())
    # if pixel is close to the copper color, set as 1, otherwise 0
    img = np.where(np.sum(np.abs(img_raw - gcolor[None, None]), axis=-1) < 1, 1, 0)

    # flip the length axis and transpose, this puts the origin at the lower left corner, and puts the
    # width axis first
    img = np.flip(img, axis=0).T

    # get board dimensions
    width = conv.in_mm(float(gerber.get_info().width_mm))
    height = conv.in_mm(float(gerber.get_info().height_mm))

    nx, ny = img.shape

    # size of each pixel along both axis
    pxl_len_x = width / nx
    pxl_len_y = height / ny

    if origin is None:
        origin = (0, 0)

    if len(origin) != 2:
        raise ValueError("Gerber file origin must be a length 2 tuple (x/y location of lower left corner.)")
    
    # location of center of first and last pixel along x
    xmin = origin[0]
    xmax = origin[0] + (width) - (pxl_len_x / 2)
    # location of center of first and last pixel and y
    ymin = origin[1]
    ymax = origin[1] + (height) - (pxl_len_y / 2)

    # create labeled array with physical coordinates of image. The coordinates are at the center of each
    # pixel.
    im_coords_x = np.linspace(xmin, xmax, nx)
    im_coords_y = np.linspace(ymin, ymax, ny)

    img = ldarray(
        img, coords=dict(x=im_coords_x, y=im_coords_y, idx_precision=dict(x=pxl_len_x, y=pxl_len_y))
    )

    return img


def plot_gerber(filepath: Path, origin: tuple = None, axes: Axes = None):
    """
    Plot a gerber file with interactive markers showing the physical coordinates. 
    Conductive areas are shown in black, open areas are shown in white.

    Parameters
    ----------
    filepath : Path
        filepath to gerber file
    origin : tuple, optional
        length 2 tuple of the x/y position of the lower left corner of the image
    """

    if origin is None:
        origin = (0, 0)

    if axes is None:
        _, axes = plt.subplots()

    image = get_gerber_image(filepath, origin)

    axes.imshow(image.T, origin="lower", cmap="binary")
    x_idx = int(len(image.coords["x"]) / 2)
    y_idx = int(len(image.coords["y"]) / 2)
    axes.set_xticks([])
    axes.set_yticks([])
    
    # add markers with labels showing the grid coordinates
    mplm.axis_marker(
        x=x_idx, 
        y=y_idx, 
        xline=dict(color="red"),
        yline=dict(color="red"),
        xformatter=lambda x: "{:.4f}".format(image.coords["x"][int(x)]), 
        yformatter=lambda y: "{:.4f}".format(image.coords["y"][int(y)]),
        axes=axes
    )
    
    return axes


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
