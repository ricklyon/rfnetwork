
from pathlib import Path
import io
import numpy as np
from scipy.optimize import fsolve
from np_struct import ldarray
import mpl_markers as mplm
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pyvista as pv
import itertools

from PIL import Image

import pygerber.gerberx3.api.v2 as pygb

# from . import conv

def blend_cell_widths(
    a: float, b: float, d: float, n_min: int = 1, tol: float = 0.0001, dtype_=np.float32
):
    """
    Return a list of cell widths that divide a cell with width d into cells that do not exceed the given
    growth rate r_max. a is the width of the previous cell, b is the width of the next cell.
    """

    # if d is less than or equal to a, no grading is possible, return d as the cell width
    if np.abs(d - a) < tol:
        return np.array([d])
    
    # swap a and b if a is larger than b. Always start with the smaller cell and increase up to b
    flip_a_b = False
    if (b < a):
        a, b = b, a
        flip_a_b = True

    # number of cells can be no larger than d/a (with all cells are of width a)
    n_max = int(d / a) + 1
    
    # if cells are nearly equal, use symmetrical grading
    symmetric_grading = ((b / a) < 1.2) and d > (2 * a)
    
    if symmetric_grading:
        # growth rates for each case where the cell is divided into n cells
        n_test = [n for n in range(2, n_max + 1) if n % 2 == 0]

        def compute_d(m, n):
            return a * np.sum([m**r for r in range(1, int(n / 2) + 1)]) - (d / 2)
    else:
        # growth rates for each case where the cell is divided into n cells
        n_test = np.arange(n_min, n_max + 1)

        def compute_d(m, n):
            return a * np.sum([m**r for r in range(1, n+1)]) - d

    # initialize arrays for the optimal growth rate m for each n
    m_result = np.zeros((len(n_test), 2))
    cells_m = [None] * len(n_test)

    for i, n in enumerate(n_test):

        m = fsolve(compute_d, x0=1.2, args=(n))[0]
        
        # if using symmetrical grading, the optimizer found the growth rate that reaches half of the cell width,
        # create the full subcell vector
        if symmetric_grading:
            cells_half = np.array([m**r for r in range(1, int(n / 2) + 1)])
            cells_m[i] = a * (np.concatenate([cells_half, np.flip(cells_half)]))
        # if using linear grading, create the list of cells with the growth rate m
        else:
            cells_m[i] = a * np.array([m**r for r in range(1, n+1)]).flatten()
        
        m_result[i] = m, b / cells_m[i][-1]

    # growth rates will decrease as n increases, so maximizing the growth rate will produce the smallest number of cells.
    # Find the growth rate the yields the fewest number of cells, while keeping the growth rate (for both internal 
    # sub-cells and between the last sub-cell and the next cell b) as near to 1.5 as possible. 
    m_err = np.abs(m_result - np.array([1.5, 1])) # np.argsort(np.abs(m_result[:, 0] - 1.5), axis=0)

    # get the m with the lowest combined difference from the desired growth rate of 1.5
    n_best_idx = np.argsort(np.sum(m_err, axis=1))[0]
    cells_best = np.array(cells_m[n_best_idx])

    # check that sub-cell widths add up the desired total width
    if np.abs(np.sum(cells_best) - d) < tol:
        return np.flip(cells_best).astype(dtype_) if flip_a_b else cells_best
    else:
        raise RuntimeError("Mesh did not converge.")
    

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

    # remove edges with zero length
    edge_len = np.sqrt(np.sum(np.abs(np.diff(obj_edges, axis=1))**2, axis=-1)).squeeze()
    obj_edges = obj_edges[edge_len > tolerance]

    # base point (or starting point) of each vector
    p1 = obj_edges[:, 0]
    # end point of each vector
    p2 = obj_edges[:, 1]

    # vector from first point to second point of edges
    v_p1_p2 = p2 - p1
    # length of each vector
    len_p1_p2 = np.clip(np.linalg.norm(v_p1_p2, axis=-1), 1e-12, None)

    # non-overlapping segments
    nonovl_edges = []

    i = np.argwhere(np.max(abs(obj_edges[..., 0] - -0.27), axis=-1) < 1e-2)
    edge = obj_edges[i]

    for i, edge in enumerate(obj_edges):

        # two end points of current edge
        A, B = edge
        v_AB = v_p1_p2[i]
        len_AB = len_p1_p2[i]

        # vectors from the base point of the current edge to all other end points from edges
        v_A_p1 = p1 - A
        v_A_p2 = p2 - A

        # length of each vector
        len_A_p1 = np.clip(np.linalg.norm(v_A_p1, axis=-1), zero, None)
        len_A_p2 = np.clip(np.linalg.norm(v_A_p2, axis=-1), zero, None)

        # dot product of the p1 and p2 vectors with all other edge vectors. Dot product is normalized by the lengths 
        # to get cos(theta). 
        p1_dot = np.einsum("ij,j->i", v_A_p1, v_AB) / (len_A_p1 * len_AB)
        p2_dot = np.einsum("ij,j->i", v_A_p2, v_AB) / (len_A_p2 * len_AB)

        # If the edges are colinear cos(theta) is 1 or -1. 
        # if the edges share a point with A, the dot product will be zero since v_A_p1 or v_A_p2 has zero length. 
        # Set to 1 to indicate the edge is in the same direction as AB.
        p1_dot = np.where(len_A_p1 <= zero * 2, 1, p1_dot)
        p2_dot = np.where(len_A_p2 <= zero * 2, 1, p2_dot)

        # set lengths to negative if lines in opposite direction
        len_A_p1 = np.sign(p1_dot) * len_A_p1
        len_A_p2 = np.sign(p2_dot) * len_A_p2

        colinear_p1 = np.abs((np.abs(p1_dot) - 1)) < tolerance
        colinear_p2 = np.abs((np.abs(p2_dot) - 1)) < tolerance

        # AB overlaps with P12 if the vector length between A -> P1 or A -> P2 is less than the length of AB, 
        # and if A -> P1 or A -> P2 makes a positive angle with AB. 
        is_edge_colinear = (colinear_p1 & colinear_p2)

        np.argwhere(is_edge_colinear)

        does_edge_overlap = is_edge_colinear & (
            ((len_A_p1 + tolerance > 0) & ((len_A_p1 - tolerance) < len_AB)) | # if p1 falls on AB
            ((len_A_p2 + tolerance > 0) & ((len_A_p2 - tolerance) < len_AB)) | # if p2 falls on AB
            (np.sign(len_A_p2) != np.sign(len_A_p1)) # p1 is on opposite side of A as p2, or vice versa
        )

        # for each edge that overlaps with AB (excluding itself) get edge indices that overlap with AB
        does_edge_overlap[i] = False
        ovl_idx = np.atleast_1d(np.argwhere(does_edge_overlap).squeeze())
        # if no other edge overlaps with AB, add the full AB edge as a non-overlapping edge
        if not len(ovl_idx):
            nonovl_edges.append((A, B))
            continue
        
        # distances from point A that define colinear line segments that overlap with AB.
        segments = []
        # for each overlapping edge with AB, get the segments along AB that overlap
        for j in ovl_idx:
            # end points of edge
            P1 = p1[j]
            P2 = p2[j]

            # if P1 is on A, overlapping segment is A->P2 if P2
            if (abs(len_A_p1[j]) < tolerance):
                segments.append((0, len_A_p2[j]))
            elif (abs(len_A_p2[j]) < tolerance):
                segments.append((0, len_A_p1[j]))
            # if A->P1 and A->P2 are in the same direction as AB, and both are smaller in length than AB, the
            # overlapping segment is fully contained in A and B
            elif (len_A_p1[j] > 0) and (len_A_p1[j] > 0) and (len_A_p1[j] < len_AB) and (len_A_p2[j] < len_AB):
                if len_A_p1[j] < len_A_p2[j]:
                    segments.append((len_A_p1[j], len_A_p2[j]))
                else: # if len_A_p1[j] < len_A_p2[j] 
                    segments.append((len_A_p2[j], len_A_p1[j]))
            # if A->P1 is in the opposite direction, P2 is before or past B, and the overlapping segment is A->P2
            elif len_A_p1[j] < 0:
                segments.append((0, len_A_p2[j]))
            # A->P2 is in the opposite direction, P1 is before or past B, and the overlapping segment is A->P1
            elif len_A_p2[j] < 0:
                segments.append((0, len_A_p1[j]))
            # both A->P1 and A->P2 are in the same direction, but one is longer than AB. 
            # if A->P1 is smaller than A->P2, the overlapping segment is P1->B
            elif len_A_p1[j] < len_A_p2[j]:
                segments.append((len_A_p1[j], len_AB))
            # A->P2 is smaller than A->P1, then A->P1 is longer than AB, overlapping segment is P2->B
            else:
                segments.append((len_A_p2[j], len_AB))

        # start with the full segment A->B, and remove the sections in segments. Each segment might
        # split the non-overlapping parts into two different parts.
        nonovl_segments = [(0, len_AB)]

        for i, seg in enumerate(segments):
            for k, nseg in enumerate(nonovl_segments.copy()):
                # clip endpoints of segment to the non-overlapping segment
                s0, s1 = np.clip(seg[0], *nseg), np.clip(seg[1], *nseg)
                # start points are the same, non-overlapping part is from end point of segment to the end
                if abs(s0 - nseg[0]) < tolerance:
                    nonovl_segments[k] = (s1, nseg[1])
                # endpoints are the same, non-overlapping part is up to the start point of segment
                elif abs(s1 - nseg[1]) < tolerance:
                    nonovl_segments[k] = (nseg[0], s0)
                # segment is between non-overlapping endpoints, split into two segments
                else:
                    nonovl_segments[k] = (nseg[0], s0)
                    nonovl_segments.append((s1, nseg[1]))
                

        # vector that when multiplied by the length of a segment, gives the vector along the line from A->B.
        # To see this for the point B,
        # B = A + v * len_AB = A + ((B - A) / len_AB) * len_AB = B
        v = (B - A) / len_AB

        # create edges from segments
        for (s0, s1) in nonovl_segments:
            # remove segments with 0 length.
            if (s1 - s0) < tolerance:
                continue

            nonovl_edges.append([(A + v * s0), (A + v * s1)])

    # for e in np.array(nonovl_edges):
    #     plt.plot(e[:, 0], e[:, 1])

    return np.array(nonovl_edges)


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
    
    in_face = np.zeros((n_faces,) + points.shape[:-1], dtype=np.int64)
    # each face in the object should be a simple polygon (no cutouts) where the following algorithm is used to 
    # determine if a point is within any face of the surface. 
    # Lines are drawn from each polygon vertex to the test point. Points that are inside will force these lines
    # to intersect an edge of the face. Points outside will cause at least one of the these lines to only meet
    # the polygon at the vertex and not touch any other edge.
    for f in range(n_faces):

        face_vertices = vertices[f]
        face_edges = np.array(edges[f])

        # fig, ax = plt.subplots(figsize=(8, 8))
        # for e in np.array(edges[f]):
        #     ax.plot(e[:, 0], e[:, 1], color="k")

        # ax.plot(points[0][0], points[0][1], marker="X", markersize=20)

        # number of intersections the edges make with lines from the object vertices to the point
        is_point_within_vertex = np.zeros((len(face_vertices),) + points.shape[:-1], dtype=np.uint8)

        # for each vertices in the closed face
        for i, v in enumerate(face_vertices):

            # get single value "x/y" coordinates for vertex
            vx, vy = v[c1_axis], v[c2_axis]

            # get the two edges that meet this vertex
            edges_x, edges_y = face_edges[..., c1_axis], face_edges[..., c2_axis]
            # check if both x and y of edge points are on vertex, if either end point is on vertex on a given
            # mark edge as True.
            is_edge_on_vertex = np.any((np.abs(edges_x - vx) < _zero_tol) & (np.abs(edges_y - vy) < _zero_tol), axis=1)
            # select the two edges with a point on the vertex
            edge_on_vertex_idx = np.argwhere(is_edge_on_vertex).squeeze()

            if len(edge_on_vertex_idx) != 2:
                raise RuntimeError(f"{len(edge_on_vertex_idx)}")
            
            # get the endpoints of the edges, the other endpoint is the vertex
            edge_1 = face_edges[edge_on_vertex_idx[0]]
            edge_2 = face_edges[edge_on_vertex_idx[1]]

            endpoint1_idx = np.argmax(np.linalg.norm((v - edge_1), axis=-1))
            endpoint2_idx = np.argmax(np.linalg.norm((v - edge_2), axis=-1))

            p1 = edge_1[endpoint1_idx]
            p2 = edge_2[endpoint2_idx]

            # vectors from vertex to the endpoint of each line attached to vertex
            v_p1 = (p1 - v)
            v_p2 = (p2 - v)

            # angle between the two lines
            len_v_p1 = np.clip(np.linalg.norm(v_p1), _zero_tol, None)
            len_v_p2 = np.clip(np.linalg.norm(v_p2), _zero_tol, None)
            vertex_ang = np.arccos(np.clip(np.dot(v_p1, v_p2) / (len_v_p1 * len_v_p2), -1, 1))

            # compute angle with every test point in the grid and the two edges
            v_tp = points - v
            len_tp_v = np.clip(np.linalg.norm(v_tp, axis=-1), _zero_tol, None)

            e1_tp_dot = np.einsum("...i, i->...", v_tp, v_p1)
            e2_tp_dot = np.einsum("...i, i->...", v_tp, v_p2) 
            e1_ang = np.arccos(np.clip(e1_tp_dot / (len_v_p1 * len_tp_v), -1, 1))
            e2_ang = np.arccos(np.clip(e2_tp_dot / (len_v_p2 * len_tp_v), -1, 1))

            # sum the angles between both edges
            tot_ang = e1_ang + e2_ang

            # if the total angle is the same as the vertex angle, the point is between the two edge lines.
            # this doesn't ensure the point is inside the face yet because the point must be between all vertex
            # angles, not just this one.
            is_point_within_vertex[i] = (
                (np.abs(tot_ang - vertex_ang) < tolerance)
            )

            # include point in surface if it is very close to a vertex. These can suffer from numerical errors when
            # computing angles.
            if d_min is not None:
                is_point_within_vertex[i] |= (np.linalg.norm(points - v, axis=-1) < (d_min / 3))


        # point is in the face if it is within the edge lines from all vertices
        in_face[f] = (
            np.all(is_point_within_vertex, axis=0) & 
            # check that all points are in the same plane as the surface (obj is a surface so all points have
            # the same value along axis)
            (np.abs(points[..., axis] - obj.points[0, axis]) < _zero_tol)
        )

    # return True if point is contained in any face of the surface
    return (np.sum(in_face, axis=0) > 0)


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