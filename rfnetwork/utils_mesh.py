
from pathlib import Path
import io
import numpy as np
from scipy.optimize import fsolve
from np_struct import ldarray
import mpl_markers as mplm
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pyvista as pv

from PIL import Image

import pygerber.gerberx3.api.v2 as pygb

from . import conv

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
    

def get_object_edges(obj: pv.PolyData, group_faces: bool = True) -> list:
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
        faces[0] = obj_edges

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
                
                if group_faces or (face_idx < 0):
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

    return faces if group_faces else faces[0]


def remove_interior_edges(obj_edges: np.ndarray, tolerance=1e-3):
    """
    Remove interior edges returned from get_object_edges(..., group_faces=False) that share an edge with other faces. 
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

    for i, edge in enumerate(obj_edges):

        # two end points of current edge
        A, B = edge
        v_AB = v_p1_p2[i]
        len_AB = len_p1_p2[i]

        # vectors from the base point of the current edge to all other end points from edges
        v_A_p1 = p1 - A
        v_A_p2 = p2 - A

        # length of each vector
        len_A_p1 = np.clip(np.linalg.norm(v_A_p1, axis=-1), 1e-12, None)
        len_A_p2 = np.clip(np.linalg.norm(v_A_p2, axis=-1), 1e-12, None)

        # dot product of the p1 and p2 vectors with all other edge vectors. Dot product is normalized by the lengths 
        # to get cos(theta). 
        p1_dot = np.einsum("ij,j->i", v_A_p1, v_AB) / (len_A_p1 * len_AB)
        p2_dot = np.einsum("ij,j->i", v_A_p2, v_AB) / (len_A_p2 * len_AB)

        # If the edges are colinear cos(theta) is 1 or -1. 
        # if the edges share a point, the dot product will be zero since v_A_p1 or v_A_p2 has zero length. 
        colinear_p1 = (np.abs((np.abs(p1_dot) - 1)) < tolerance) | (len_A_p1 < tolerance)
        colinear_p2 = (np.abs((np.abs(p2_dot) - 1)) < tolerance) | (len_A_p2 < tolerance)

        # AB overlaps with P12 if the vector length between A -> P1 or A -> P2 is less than the length of AB, 
        # and if A -> P1 or A -> P2 makes a positive angle with AB. 
        does_edge_overlap = (
            (colinear_p1 & colinear_p2) & 
            (((len_A_p1 - tolerance) < len_AB) | ((len_A_p2 - tolerance) < len_AB)) &
            ((p1_dot > 0) | (p2_dot > 0))
        )

        # for each edge that overlaps with AB (excluding itself)
        does_edge_overlap[i] = False

        # points on AB that defines a non-overlapping segment from A->Ao and B->Bo
        Ao, Bo = [], []

        # get edges that overlap with AB
        ovl_idx = np.atleast_1d(np.argwhere(does_edge_overlap).squeeze())

        if not len(ovl_idx):
            nonovl_edges.append((A, B))
            continue
        
        # for each overlapping edge with AB
        for j in ovl_idx:
            # end points of edge
            P1 = p1[j]
            P2 = p2[j]

            # if A->P1 and A->P2 are in the same direction as AB, and both are smaller in length than AB, this splits 
            # the non-overlapping part of AB into two parts.
            if p1_dot[j] > 0 and p2_dot[j] > 0 and (len_A_p1[j] < len_AB) and (len_A_p2[j] < len_AB):
                if len_A_p1[j] < len_A_p2[j]:
                    Ao.append(P1)
                    Bo.append(P2)
                else: # if len_A_p1[j] < len_A_p2[j] 
                    Ao.append(P2)
                    Bo.append(P1)
            # if A->P1 is in the opposite direction, P2 is on AB, and the non-overlapping segment is P2->B
            elif p1_dot[j] < 0:
                Bo.append(P2)
                Ao.append(A)
            # A->P2 is in the opposite direction, P1 is on AB, and the non-overlapping segment is P1->B
            elif p2_dot[j] < 0:
                Bo.append(P1)
                Ao.append(A)
            # both A->P1 and A->P2 are in the same direction, but one is longer than AB. 
            # if A->P1 is smaller than A->P2, the non-overlapping segment is A->P1
            elif len_A_p1[j] < len_A_p2[j]:
                Ao.append(P1)
                Bo.append(B)
            # A->P2 is smaller than A->P1, then A->P is longer than AB, overlapping segment is A->P2
            else:
                Ao.append(P2)
                Bo.append(B)

        # compute distances of A->Ao and B->Bo. Keep the points with the smallest distance as B->Bo defines
        # a segment that was not overlapped by any of the other edges.
        len_AAo = np.linalg.norm(np.array(Ao) - A, axis=-1)
        len_BBo = np.linalg.norm(np.array(Bo) - B, axis=-1)
        Ao_idx, Bo_idx = np.argmin(len_AAo), np.argmin(len_BBo)

        # add non-overlapping line segments to the list
        if len_AAo[Ao_idx] > tolerance:
            nonovl_edges.append((A, Ao[Ao_idx]))

        if len_BBo[Bo_idx] > tolerance:
            nonovl_edges.append((B, Bo[Bo_idx]))

    # for e in np.array(nonovl_edges):
    #     plt.plot(e[:, 0], e[:, 1])


    return np.array(nonovl_edges)




def get_object_vertices(obj: pv.PolyData, group_faces: bool = True) -> list:
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

            if group_faces:
                face_idx += 1

    return faces if group_faces else faces[0]


def is_point_in_surface(points, obj, tolerance=0.001):
    """
    Returns an array the same shape as point with each value set to True if point is inside the 2D surface.

    """

    # tolerance for zero values
    _zero_tol = 1e-5

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

    # return all zeros if object mesh is empty
    if not len(obj.points):
        return np.zeros(points.shape[:-1])
    
    vertices = get_object_vertices(obj, group_faces=True)
    edges = get_object_edges(obj, group_faces=True)
    n_faces = len(edges)

    in_face = np.zeros((n_faces,) + points.shape[:-1], dtype=np.int64)
    # for each face in the object
    for f in range(n_faces):

        face_vertices = vertices[f]
        face_edges = edges[f]

        # fig, ax = plt.subplots(figsize=(8, 8))
        # for e in np.array(edges[f]):
        #     ax.plot(e[:, 0], e[:, 1], color="k")

        # ax.plot(points[0][0], points[0][1], marker="X", markersize=20)

        # number of intersections the edges make with lines from the object vertices to the point
        n_edge_intersections = np.zeros((len(face_vertices),) + points.shape[:-1], dtype=np.int64)

        for i, v in enumerate(face_vertices):

            for edge in face_edges:

                edge = np.array(edge)
                
                # skip if base point is on either end point of the edge
                if np.any(np.sum(np.abs(edge - v[None]), axis=1) < _zero_tol):
                    continue

                # get "x/y" coordinates for vertices
                x1, y1 = v[c1_axis], v[c2_axis]

                # get "x/y" coordinates for test points
                x2, y2 = points[..., c1_axis], points[..., c2_axis]

                with np.errstate(divide='ignore', invalid='ignore'):
                    # slope of edge line and line from the object vertices to the test point
                    # m = (y2 - y1) / (x2 - x1)
                    m1 = (y2 - y1) / (x2 - x1)
                    m2 = (edge[1, c2_axis] - edge[0, c2_axis]) / (edge[1, c1_axis] - edge[0, c1_axis])

                    # y intercept point
                    b1 = y1 - m1 * x1
                    b2 = edge[0, c2_axis] - m2 * edge[0, c1_axis]

                    # intersection coordinate.
                    # if m1 is infinite, the line is vertical and the x intersection coordinate is just the x
                    # coordinate of the line.
                    x_int = np.where(
                        ~np.isfinite(m1), x1, np.where(~np.isfinite(m2), edge[0, c1_axis], (b2 - b1) / (m1 - m2))
                    )
                    # plug x_int into the line equation with a finite slope to get the y intersection point
                    y_int = np.where(np.isfinite(m1), m1 * x_int + b1, m2 * x_int + b2)

                # increment if the vertex line and edge intersect
                n_edge_intersections[i] += (
                    ((x_int - tolerance) <= np.max(edge[:, c1_axis])) & ((x_int + tolerance) >= np.min(edge[:, c1_axis])) &
                    ((y_int - tolerance) <= np.max(edge[:, c2_axis])) & ((y_int + tolerance) >= np.min(edge[:, c2_axis]))
                )
                
                # ax.plot(x_int, y_int, marker="o")
                # ax.set_xlim([p0[0], p1[0]])
                # ax.set_ylim([p0[1], p1[1]])

        # point is in the face if the line from each vertices intersects with an edge 
        in_face[f] = (
            np.all(n_edge_intersections, axis=0) & 
            (np.abs(points[..., axis] - obj.points[0, axis]) < _zero_tol)
        )

        # if point is far away from the face, lines become nearly parallel and tolerances can lead to false
        # intersections. Correct points that are wholly outside the face bounding box
        f_pmin = np.min(face_vertices, axis=0)
        f_pmax = np.max(face_vertices, axis=0)
        x0, y0 = f_pmin[c1_axis], f_pmin[c2_axis]
        x1, y1 = f_pmax[c1_axis], f_pmax[c2_axis]

        xp, yp = points[..., c1_axis], points[..., c2_axis]

        bbox_in_shape = (
            ((xp - tolerance) <= x1) & ((xp + tolerance) >= x0) &
            ((yp - tolerance) <= y1) & ((yp + tolerance) >= y0)
        )

        in_face[f] = np.where(bbox_in_shape, in_face[f], 0)

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