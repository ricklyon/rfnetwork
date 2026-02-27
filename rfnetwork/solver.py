import itertools
import time
import numpy as np 
import pyvista as pv
from copy import copy
from np_struct import ldarray
import sys

from rfnetwork import const, conv, utils, core

u0 = const.u0
e0 = const.e0
c0 = const.c0


class FDTD_Solver():
    """
    FDTD EM Solver for PCB geometries.
    """

    def __init__(self, bounding_box):

        self.bounding_box = bounding_box
        self.dielectric = dict()
        self.conductor = dict()
        self.lumped_element = dict()

        self.pml_boundaries = []

        self.monitors=dict()
        self.probes=dict()
        
        self.n_pml = None
        self._auto_name_counter = 0

        self.sbox_max = np.max(bounding_box.points, axis=0)
        self.sbox_min = np.min(bounding_box.points, axis=0)

        self._places = 5 # hundredth of a mil
        self._tol = 1 / (10 ** self._places)

        self._meshed = False
        self._solved = False

    def invalidate_mesh(self):
        self._meshed = False
        self.invalidate_solution()

    def invalidate_solution(self):
        self._solved = False

    def check_mesh(self):
        """ Raise error if model is not meshed yet. """

        if not self._meshed:
            raise RuntimeError("Mesh has been invalidated or has not been created yet.")
        
    def check_solution(self):
        """ Raise error if model is not solved yet. """
        
        if not self._solved:
            raise RuntimeError("Solution has been invalidated or model has not been solved yet.")

    def _add_object(self, group, objects, properties, name: str = None):
        """
        """
        self.invalidate_mesh()

        # cast as iterable 
        if not isinstance(objects, (tuple, list, np.ndarray)):
            objects = [objects]

        # check that name is not used already
        if name is not None and name in self.conductor.keys():
            raise ValueError(f"Duplicate conductor name: {name}")

        for i, obj in enumerate(objects):
            
            # increment name counter for each object
            if name is None:
                obj_name = f"obj_{self._auto_name_counter}"
                self._auto_name_counter += 1
            else:
                obj_name = name + "_{i}" if len(objects) > 1 else name

            group[obj_name] = dict(obj=obj, **properties)

    def add_dielectric(self, *objects, er: float, loss_tan=0, f0=0, style: dict = dict(), name: str = None):
        """
        Add rectangular dielectric
        """
        self._add_object(
            self.dielectric, objects, properties=dict(er=er, loss_tan=loss_tan, f0=f0, style=style), name=name
        )

    def add_conductor(self, *objects, sigma=1e16, style: dict = dict(), name: str = None):
        """
        Add conductor
        """
        self._add_object(self.conductor, objects, properties=dict(sigma=sigma, style=style), name=name)

    def add_lumped_port(self, number, face, integration_axis: str, r0: float = 50, name: str = None):
        """
        Attach a lumped port to a face
        """
        self._add_object(
            self.lumped_element, 
            face, 
            properties=dict(integration_axis=integration_axis, r=r0, port=number), 
            name=name
        )

    def add_resistor(self, face, value: float, integration_axis: str, name: str = None):
        """
        Attach a resistive element to a face
        """
        
        self._add_object(
            self.lumped_element, 
            face, 
            properties=dict(integration_axis=integration_axis, r=value, port=None), 
            name=name
        )

    def assign_PML_boundaries(self, *sides: str, n_pml: float = 10):
        """
        Assign a PML boundary to sides of the solve box. All sides must have the same number of PML cells.

        Parameters
        ----------
        sides : list, str
            Valid values are ("x+", "x-", "y+", "y-", "z+", "z-",)
        n_pml : int, default: 10
            number of PML cells.
        """
        self.invalidate_mesh()

        valid_sides = ("x+", "x-", "y+", "y-", "z+", "z-")
        if any([s not in valid_sides for s in sides]):
            raise ValueError(f"PML side not recognized. Expecting one of {valid_sides}")

        self.pml_boundaries = copy(list(sides))
        self.n_pml = n_pml

    def pos_to_idx(self, p, mode: str = "edge"):
        """
        Returns the index of the grid edge or cell center that is directly on or just past (in +x, +y and +z directions) 
        the given position.
        """
        self.check_mesh()
        idx = []
        mode = [mode] * 3 if isinstance(mode, str) else mode
        
        grid = [self.g_edges[i] if m == "edge" else self.g_cells[i] for i, m in enumerate(mode)]
        for i, g in enumerate(grid):
            diff = (g - p[i])
            # if no cell is above the point, return the length of the axis, otherwise return the first cell that is 
            # larger than the point.
            idx += [np.argmax(diff >= -self._tol) if diff[-1] > -self._tol else len(g)]

        return tuple(idx)

    def field_pos_to_idx(self, position, field: str):
        """
        Returns the index of the field component that is directly on or just past (in +x, +y and +z directions) 
        the given grid position.
        """

        self.check_mesh()
        
        floc = self.floc[field]

        idx = []
        for i, g in enumerate(floc):
            diff = (g - position[i])
            # if no cell is above the point, return the length of the axis, otherwise return the first cell that is 
            # equal to or larger than the point.
            idx += [np.argmax(diff >= -self._tol) if diff[-1] > -self._tol else len(g)]
        
        return tuple(idx)
    
    def generate_mesh(self, d0: float, d_edge: float = None):
        """
        Generate the grid and FDTD for the model geometry.

        Parameters
        ----------
        d0 : float
            nominal cell width, inches. This cell width is used in open regions far from geometry features.
        d_edge : float, optional
            cell width 
        """

        self._meshed = True
        self._solved = False

        self._init_grid(d0=d0, d_edge=d_edge)
        # init dielectrics sets the cell sigma and er but does not set the coefficients
        self._init_dielectrics()
        # initialize coefficients from the cell sigma and er set by the dielectrics
        self._init_coefficients()

        # add PML layers before conductors and ports, this gives priority to any conductors that may extend into the 
        # PML
        for pml_side in self.pml_boundaries:
            self._init_PML(pml_side)

        self._init_conductors()
        self._init_lumped_elements()

        
    def get_object_edges(self, obj, group_faces: bool = True) -> list:
        """
        Get the endpoints of the lines that form the edges of the object faces. Returns a M-length list for each 
        face in the object, each containing a N-length list of 2x3 arrays. N is the number of edges in the face.
        Each row is the coordinates of the two endpoints of the edge.

        Parameters
        ----------
        group_faces: bool, default: True
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


    def get_object_vertices(self, obj, group_faces: bool = True) -> list:
        """
        Get the vertices of each face of the object. Returns a M-length list for each face in the
        object, each containing a Nx3 array of the vertices coordinates in that face.

        Parameters
        ----------
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


    def is_point_in_surface(self, points, obj, tolerance=0.0005):
        """
        Returns an array the same shape as point with each value set to True if point is inside the 2D surface
        """

        points = np.atleast_2d(points)

        # object bounding box
        p0 = np.min(obj.points, axis=0)
        p1 = np.max(obj.points, axis=0)

        # get axis normal to the surface
        normal_axis = (np.diff([p0, p1], axis=0) == 0)
        axis = np.argmax(normal_axis)

        if not np.any(np.abs(normal_axis) > self._tol):
            raise ValueError("Object must be a surface on a cardinal plane.")
        
        # get the two axis of the component vectors on the surface
        c1_axis, c2_axis = ((1, 2), (0, 2), (0, 1))[axis]

        # return all zeros if object mesh is empty
        if not len(obj.points):
            return np.zeros(points.shape[:-1])
        
        vertices = self.get_object_vertices(obj, group_faces=True)
        edges = self.get_object_edges(obj, group_faces=True)
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
                    if np.any(np.sum(np.abs(edge - v[None]), axis=1) < self._tol):
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
                (np.abs(points[..., axis] - obj.points[0, axis]) < self._tol)
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


    def _get_mesh_points(self, d_edge: float):
        """
        Compile all points defined by model geometry, as well as edge cells around geometry features.
        """

        # conductor objects
        objects = [cond["obj"] for cond in self.conductor.values()]

        # substrate objects
        sub_objects = [self.bounding_box] + [sub["obj"] for sub in self.dielectric.values()]

        # get a list of 2x3 arrays, coordinates of each endpoint of the edges of the model objects
        obj_edges = []
        for obj in objects:
            obj_edges += self.get_object_edges(obj, group_faces=False)

        obj_edges = np.around(obj_edges, decimals=self._places).astype(np.float32)

        # object vertices
        hard_points = [np.unique(obj_edges[..., i].flatten()) for i in range(3)]

        # break angled edges up into sections
        soft_points = [np.array([]), np.array([]), np.array([])]
        
        # object vertices and points for the mesh around angled sections
        all_points = [np.array([]), np.array([]), np.array([])]

        edge_len = np.abs(np.diff(obj_edges, axis=1).squeeze())
        # compute the area of a box bound by this edge and the cardinal axis. If area is above a certain threshold
        # related to d_edge, create soft points along the axis to keep the area below the threshold. 
        edge_area = np.prod(edge_len[:, :2], axis=-1)

        # add small edge cells around object transitions to reduce error
        for i, edge in enumerate(obj_edges):
            if edge_area[i] > (d_edge ** 2):
                # break angled edges along x and y into sub-cells separated by d_edge
                nx_ny = np.around(np.abs(edge_len[i][:2]) / d_edge).astype(int)

                for axis in range(2):
                    soft_points[axis] = np.append(
                        soft_points[axis], np.linspace(edge[0, axis], edge[1, axis], nx_ny[axis])
                    )
            
            # add edge cells on either side of edges aligned with the cardinal axis
            elif np.abs(edge_area[i]) < self._tol:
                
                for axis in range(3):
                    # if edge is normal to the axis (both endpoints are at the same value), add edge cells on 
                    # either side. Skip if edge is at the bounding box edge
                    at_bbox_edge = (
                        (np.abs(edge[0][axis] - self.sbox_max[axis]) < self._tol) |
                        (np.abs(edge[0][axis] - self.sbox_min[axis]) < self._tol)
                    )

                    if (edge_len[i][axis] < self._tol) and not at_bbox_edge:
                        soft_points[axis] = np.append(
                            soft_points[axis], [edge[0][axis] - d_edge, edge[0][axis] + d_edge]
                        )


        for axis in range(3):
            # add points for substrates and bounding box
            for obj in sub_objects:
                hard_points[axis] = np.concatenate([hard_points[axis], obj.points[:, axis]])
        
            # remove any soft points that are less than d_edge away from an object point
            for p in np.unique(hard_points[axis]):
                soft_points[axis] = np.where(np.abs(p - soft_points[axis]) < (d_edge * 0.8), np.nan, soft_points[axis])
            
            # clean up and sort soft mesh points
            sp_axis = np.sort(np.unique(np.around(soft_points[axis], decimals=self._places)))
            
            # create a reduced set of mesh points that are spaced no less than d_edge. Walk through all soft points
            # and only add points to the reduced list if they are at least d_edge away from the last point.
            if len(sp_axis):
                sp_axis_reduced = [sp_axis[0]]
                last_p = sp_axis[0]
                for i, p in enumerate(sp_axis[1:]):
                    if (p - last_p) >= d_edge:
                        sp_axis_reduced += [p]
                        last_p = p

            soft_points[axis] = sp_axis_reduced

            # clip points outside of the sbox limits
            soft_points[axis] = np.clip(soft_points[axis], self.sbox_min[axis], self.sbox_max[axis])
            hard_points[axis] = np.clip(hard_points[axis], self.sbox_min[axis], self.sbox_max[axis])

            # combine object points with soft points
            all_points[axis] = np.concatenate([hard_points[axis], soft_points[axis]])
            # round to tolerance
            all_points[axis] = np.around(all_points[axis], decimals=self._places)
            # remove repeated values and sort
            all_points[axis] = np.sort(np.unique(all_points[axis]))

            # remove nan values 
            all_points[axis] = all_points[axis][np.isfinite(all_points[axis])]

        return all_points

    def _init_grid(self, d0: float, d_edge: float = None):
        """ Create model grid. """
        
        if d_edge is not None and d_edge > (d0 / 1.5):
            raise ValueError("d_edge must be less than d0.")
        
        all_points = self._get_mesh_points(d_edge=d0 if d_edge is None else d_edge)

        if not isinstance(d0, list):
            d0 = [d0] * 3

        dtype_ = np.float32

        # build a list of cell widths along each axis
        mesh_cells_d = [[], [], []]
        for axis in range(3):

            # list of distances between each edge
            cells_d = np.diff(all_points[axis])

            # cells are broken up into smaller sub-cells to create small edge cells, and to shorten the span
            # that is graded with increasingly larger cells.
            subcells_d = []

            for i, d in enumerate(cells_d):

                # split cells larger than d0*5 into multiple cells, this prevents grading over large spans
                split_threshold = d0[axis] * 3
                if int(d / split_threshold) >= 1:
                    n_split = int(d / split_threshold)
                    # split cell width into equal parts
                    subcells_d_i = [d / n_split] * n_split
                else:
                    n_split = 1
                    subcells_d_i = [d]

                subcells_d += subcells_d_i

            # cells indices arranged from largest width to smallest
            # blend large cells first so they transition gradually into smaller ones.
            # The larger cells have more room to work in and
            # it's easier to blend. The smaller cells have less work to do because the
            # larger cells have already stepped down to meet their widths.
            # Leave PEC cells for last so they don't blend up to the (typically) larger d0 cells
            d_order = np.flip(np.argsort(subcells_d))
            
            graded_subcells_d = [None] * len(subcells_d)
            for i in d_order:
                
                # current cell width
                d = subcells_d[i]

                # previous cell width
                if (i > 0) and (graded_subcells_d[i - 1] is not None):
                    dprev = graded_subcells_d[i - 1][-1]
                elif (i > 0):
                    dprev = np.clip(subcells_d[i - 1], 0, d0[axis] )
                # If on the edge of the grid, match to d0 if the cell width is near d0, otherwise match
                # to the edge width
                else:
                    dprev = d0[axis] if d >= (d0[axis] * 0.9) else d_edge

                # next cell width
                if (i < len(subcells_d) - 1) and (graded_subcells_d[i + 1] is not None):
                    dnext = graded_subcells_d[i + 1][0]
                elif (i < len(subcells_d) - 1) :
                    dnext = np.clip(subcells_d[i + 1], 0, d0[axis] )
                # If on the edge of the grid, match to the boundary width
                else:
                    dnext = d0[axis] if d >= (d0[axis] * 0.9) else d_edge

                # if cell is bounded by d0 cells on either side, divide the cell equally 
                if all([(g / d0[axis] ) > 0.8 for g in [dprev, dnext, d]]):
                    n_split = int(np.around(d / d0[axis] ))
                    graded_subcells_d[i] = [d / n_split] * n_split
                # create a gradient of cell widths to span the space that minimizes the growth rate. 
                else:
                    graded_subcells_d[i] = list(
                        utils.blend_cell_widths(dprev, dnext, d, tol=self._tol)
                    )

            # flatten list of lists of subcell widths
            mesh_cells_d[axis] = list(itertools.chain(*graded_subcells_d))

        
        gx, gy, gz = [np.around(np.concatenate([[self.sbox_min[i]], self.sbox_min[i] + np.cumsum(mesh_cells_d[i])]), decimals=self._places) for i in range(3)]
        dx, dy, dz = np.diff(gx).astype(dtype_), np.diff(gy).astype(dtype_), np.diff(gz).astype(dtype_)

        self.n_cells = len(dx), len(dy), len(dz)  
        self.grid_mesh = pv.RectilinearGrid(gx.astype(dtype_), gy.astype(dtype_), gz.astype(dtype_))

        self.Nx, self.Ny, self.Nz = self.n_cells
        self.dx, self.dy, self.dz = dx, dy, dz
        
        # locations of cell centers
        gx_h = (gx[1:] + gx[:-1]) / 2
        gy_h = (gy[1:] + gy[:-1]) / 2
        gz_h = (gz[1:] + gz[:-1]) / 2

        # half cell lengths between h components
        dx_h = (dx[1:] + dx[:-1]) / 2
        dy_h = (dy[1:] + dy[:-1]) / 2
        dz_h = (dz[1:] + dz[:-1]) / 2

        self.g_edges = gx, gy, gz
        self.g_cells = gx_h, gy_h, gz_h
        self.d_cells = dx, dy, dz
        self.dh_cells = dx_h, dy_h, dz_h
        self.dx_h, self.dy_h, self.dz_h = dx_h, dy_h, dz_h

        self.eps = np.ones(self.n_cells) * e0
        self.sigma = np.zeros(self.n_cells)

        # locations of all field components in grid
        self.floc = dict(
            ex=(gx_h, gy, gz),
            ey=(gx, gy_h, gz),
            ez=(gx, gy, gz_h),
            hx=(gx, gy_h, gz_h),
            hy=(gx_h, gy, gz_h),
            hz=(gx_h, gy_h, gz)
        )

        # field shapes
        Nx, Ny, Nz = self.n_cells
        self.fshape = dict(
            ex=(Nx, Ny+1, Nz+1),
            ey=(Nx+1, Ny, Nz+1),
            ez=(Nx+1, Ny+1, Nz),
            hx=(Nx+1, Ny, Nz),
            hy=(Nx, Ny+1, Nz),
            hz=(Nx, Ny, Nz+1)
        )

        # pad the combined cell widths
        # so edge components are assigned zero width
        dx_hp = np.pad(dx_h, (1, 1))
        dy_hp = np.pad(dy_h, (1, 1))
        dz_hp = np.pad(dz_h, (1, 1))
        # cell widths along the direction of the component
        self.fcell_w = dict(
            ex=(dx, dy_hp, dz_hp),
            ey=(dx_hp, dy, dz_hp),
            ez=(dx_hp, dy_hp, dz),
            hx=(dx_hp, dy, dz),
            hy=(dx, dy_hp, dz),
            hz=(dx, dy, dz_hp)
        )

    def _init_dielectrics(self):

        for sub in self.dielectric.values():
            if sub["obj"].n_cells > 6:
                raise NotImplementedError("Only rectangular dielectrics are supported.")
        
            x0, y0, z0 = self.pos_to_idx(np.min(sub["obj"].points, axis=0), mode="cell")
            x1, y1, z1 = self.pos_to_idx(np.max(sub["obj"].points, axis=0), mode="cell")

            # assign the max dk assigned to each cell (dielectrics with higher dk will take priority)
            self.eps[x0: x1, y0: y1, z0: z1] = np.where(
                self.eps[x0: x1, y0: y1, z0: z1] > e0 * sub["er"], 
                self.eps[x0: x1, y0: y1, z0: z1],
                e0 * sub["er"], 
            )
            # non-dispersive conductivity at a single frequency, assign max sigma to each cell
            sub_sigma = sub["loss_tan"] * e0 * sub["er"] * 2 * np.pi * sub["f0"]
            self.sigma[x0: x1, y0: y1, z0: z1] = np.where(
                self.sigma[x0: x1, y0: y1, z0: z1] > sub_sigma,
                self.sigma[x0: x1, y0: y1, z0: z1],
                sub_sigma, 
            )

    def _init_conductors(self):
        """
        Write the coefficient values for the conductors. 
        """

        # epsilon at each field component
        e_eps = dict(ex=self.eps_ex, ey=self.eps_ey, ez=self.eps_ez)

        # split component field names
        e_split = dict(ex=("ex_y", "ex_z"), ey=("ey_z", "ey_x"), ez=("ez_x", "ez_y"))

        for cond in self.conductor.values():
            obj = cond["obj"]
            sig = cond["sigma"]

            for field in ["ex", "ey", "ez"]:

                # get indices of the grid edges that bound the conductor. The mesh ensures that conductor edges fall
                # on grid edges.
                p0 = list(self.field_pos_to_idx(np.min(obj.points, axis=0), field))
                p1 = list(self.field_pos_to_idx(np.max(obj.points, axis=0), field))

                is_surface = np.any(np.diff([p0, p1], axis=0) == 0)

                # index for the bounding box of the object
                bbox_idx = tuple([slice(p0[i], p1[i] + 1) for i in range(3)])

                # get the grid locations that are inside the bounding box of the conductor
                e_grid_points = np.meshgrid(*[self.floc[field][i][bbox_idx[i]] for i in range(3)], indexing="ij")

                # point cloud of all grid points in the bounding box
                e_pdata = pv.PolyData(np.transpose(e_grid_points, axes=(1, 2, 3, 0)).reshape(-1, 3))

                if is_surface:
                    # get an array the same shape as the grid_points with zeros for points outside the object and ones 
                    # for points inside
                    points = np.transpose(e_grid_points, axes=(1, 2, 3, 0))
                    inside_mask = self.is_point_in_surface(points, obj)

                else:
                    dist = e_pdata.compute_implicit_distance(obj)
                    mesh_dist = dist["implicit_distance"].reshape(e_grid_points[0].shape)
                    inside_mask = (mesh_dist - self._tol) <= 0

                # conductor coefficients
                eps_bbox = e_eps[field][bbox_idx]
                Ca_c = (2 * eps_bbox - (sig * self.dt)) / (2 * eps_bbox + (sig * self.dt))
                Cb_c = (2 * self.dt) / ((2 * eps_bbox + (sig * self.dt)))

                for e_sp in e_split[field]:
                    self.Ca[e_sp][bbox_idx] = np.where(inside_mask, Ca_c, self.Ca[e_sp][bbox_idx])
                    self.Cb[e_sp][bbox_idx] = np.where(inside_mask, Cb_c, self.Cb[e_sp][bbox_idx])


    def _init_coefficients(self):
        """
        Build FDTD coefficients for all grid cells.
        """
        dtype_ = np.float32
        
        dx, dy, dz = [conv.m_in(d) for d in self.d_cells]
        Nx, Ny, Nz = self.n_cells
        
        # compute maximum time step that ensures convergence, use freespace propagation speed as worst case
        length_min = np.array([np.min(dx), np.min(dy), np.min(dz)], dtype=dtype_)
        dmin = 1 / np.sqrt(((1 / length_min)**2).sum())
        dt = 0.95 * (dmin / const.c0)
        self.dt = dt
        
        Ca_0 = 1  # (2 * e0 - (sig_0 * dt)) / (2 * e0 + (sig_0 * dt))
        Cb_0 = dt / e0  # (2 * dt) / ((2 * e0 + (sig_0 * dt)))
        Da_0 = 1  # (2 * u0 - (sigm_0 * dt)) / (2 * u0 + (sigm_0 * dt))
        Db_0 = dt / u0


        def average_cells(cell_values):
            # cell widths broadcasted across the full 3D grid
            dx0, dx1 = dx[:-1][..., None, None], dx[1:][..., None, None]
            dy0, dy1 = dy[:-1][None, :, None], dy[1:][None, :, None]
            dz0, dz1 = dz[:-1][None, None], dz[1:][None, None]

            # average cells adjacent to x edges
            v_x = (cell_values[:-1] * (dx0/2) + cell_values[1:] * (dx1/2)) / (dx0/2 + dx1/2)
            # average cells adjacent to y edges
            v_y = (cell_values[:, :-1] * (dy0/2) + cell_values[:, 1:] * (dy1/2)) / (dy0/2 + dy1/2)
            # average cells adjacent to z edges
            v_z = (cell_values[..., :-1] * (dz0/2) + cell_values[..., 1:] * (dz1/2)) / (dz0/2 + dz1/2)

            # average cells along y and z
            v_yz = ((v_y[..., :-1] * (dz0/2) + v_y[..., 1:] * (dz1/2)) / (dz0/2 + dz1/2))
            # average cells along x and z
            v_xz = ((v_x[..., :-1] * (dz0/2) + v_x[..., 1:] * (dz1/2)) / (dz0/2 + dz1/2))
            # average cells along x and y
            v_xy = ((v_x[:, :-1] * (dy0/2) + v_x[:, 1:] * (dy1/2)) / (dy0/2 + dy1/2))

            return v_x, v_y, v_z, v_xy, v_xz, v_yz


        eps_x, eps_y, eps_z, eps_xy, eps_xz, eps_yz = average_cells(self.eps)

        sig_x, sig_y, sig_z, sig_xy, sig_xz, sig_yz = average_cells(self.sigma)

        self.eps_ex = np.ones(self.fshape["ex"], dtype=dtype_) * e0
        self.eps_ey = np.ones(self.fshape["ey"], dtype=dtype_) * e0
        self.eps_ez = np.ones(self.fshape["ez"], dtype=dtype_) * e0

        # ex component is on the y and z edge of the cell
        self.eps_ex[:, 1:-1, 1:-1] = eps_yz
        # ey component is on the x and z edge of the cell
        self.eps_ey[1:-1, :, 1:-1] = eps_xz
        # ez component is on the x and y edge of the cell, average eps from adjacent cells on both axis
        self.eps_ez[1:-1, 1:-1] = eps_xy

        # eps at h components
        self.eps_hx = np.ones(self.fshape["hx"], dtype=dtype_) * e0
        self.eps_hy = np.ones(self.fshape["hy"], dtype=dtype_) * e0
        self.eps_hz = np.ones(self.fshape["hz"], dtype=dtype_) * e0

        # ex component is on the x edge of the cell
        self.eps_hx[1:-1, ] = eps_x
        # ey component is on the y edge of the cell
        self.eps_hy[:, 1:-1] = eps_y
        # ez component is on the x and y edge of the cell, average eps from adjacent cells on both axis
        self.eps_hz[..., 1:-1] = eps_z

        self.sig_ex = np.zeros(self.fshape["ex"], dtype=dtype_)
        self.sig_ey = np.zeros(self.fshape["ey"], dtype=dtype_)
        self.sig_ez = np.zeros(self.fshape["ez"], dtype=dtype_)

        # sigma values at ex
        self.sig_ex[:, 1:-1, 1:-1] = sig_yz
        # sigma values at ey
        self.sig_ey[1:-1, :, 1:-1] = sig_xz
        # sigma values az ez
        self.sig_ez[1:-1, 1:-1] = sig_xy

        # coefficient in front of the previous time values of E
        Ca_ex = (2 * self.eps_ex - (self.sig_ex * dt)) / (2 * self.eps_ex + (self.sig_ex * dt))
        Ca_ey = (2 * self.eps_ey - (self.sig_ey * dt)) / (2 * self.eps_ey + (self.sig_ey * dt))
        Ca_ez = (2 * self.eps_ez - (self.sig_ez * dt)) / (2 * self.eps_ez + (self.sig_ez * dt))
        
        # coefficient in front of the difference terms of H
        Cb_ex = (2 * dt) / ((2 * self.eps_ex + (self.sig_ex * dt)))
        Cb_ey = (2 * dt) / ((2 * self.eps_ey + (self.sig_ey * dt)))
        Cb_ez = (2 * dt) / ((2 * self.eps_ez + (self.sig_ez * dt)))
    
        self.Ca = dict(
            ex_y = Ca_ex.copy(),
            ex_z = Ca_ex.copy(),
            ey_z = Ca_ey.copy(),
            ey_x = Ca_ey.copy(),
            ez_x = Ca_ez.copy(),
            ez_y = Ca_ez.copy()
        )

        self.Cb = dict(
            ex_y = Cb_ex.copy(),
            ex_z = Cb_ex.copy(),
            ey_z = Cb_ey.copy(),
            ey_x = Cb_ey.copy(),
            ez_x = Cb_ez.copy(),
            ez_y = Cb_ez.copy()
        )

        self.Da = dict(
            hx_y = np.ones((Nx+1, Ny, Nz), dtype=dtype_) * Da_0,
            hx_z = np.ones((Nx+1, Ny, Nz), dtype=dtype_) * Da_0,
            hy_z = np.ones((Nx, Ny+1, Nz), dtype=dtype_) * Da_0,
            hy_x = np.ones((Nx, Ny+1, Nz), dtype=dtype_) * Da_0,
            hz_x = np.ones((Nx, Ny, Nz+1), dtype=dtype_) * Da_0,
            hz_y = np.ones((Nx, Ny, Nz+1), dtype=dtype_) * Da_0,
        )
        
        self.Db = dict(
            hx_y1 = np.ones((Nx+1, Ny, Nz), dtype=dtype_) * Db_0,
            hx_y2 = np.ones((Nx+1, Ny, Nz), dtype=dtype_) * Db_0,
            hx_z1 = np.ones((Nx+1, Ny, Nz), dtype=dtype_) * Db_0,
            hx_z2 = np.ones((Nx+1, Ny, Nz), dtype=dtype_) * Db_0,
            hy_z1 = np.ones((Nx, Ny+1, Nz), dtype=dtype_) * Db_0,
            hy_z2 = np.ones((Nx, Ny+1, Nz), dtype=dtype_) * Db_0,
            hy_x1 = np.ones((Nx, Ny+1, Nz), dtype=dtype_) * Db_0,
            hy_x2 = np.ones((Nx, Ny+1, Nz), dtype=dtype_) * Db_0,
            hz_x1 = np.ones((Nx, Ny, Nz+1), dtype=dtype_) * Db_0,
            hz_x2 = np.ones((Nx, Ny, Nz+1), dtype=dtype_) * Db_0,
            hz_y1 = np.ones((Nx, Ny, Nz+1), dtype=dtype_) * Db_0,
            hz_y2 = np.ones((Nx, Ny, Nz+1), dtype=dtype_) * Db_0,
        )

            
    def _init_lumped_elements(self):
        """
        
        """
        # get the lumped element faces that are assigned as port terminations
        port_faces = [element for element in self.lumped_element.values() if element["port"] is not None]
        self.ports = [None] * len(port_faces)
        
        d_cells = [conv.m_in(d) for d in self.d_cells]
        dh_cells = [conv.m_in(d) for d in self.dh_cells]
        
        # element = self.lumped_element["obj_2"]
        for name, element in self.lumped_element.items():

            face = element["obj"]
            # total resistance
            r = element["r"]
            # voltage integration direction 
            integration_axis = element["integration_axis"]
            port = element["port"]
    
            if face.n_cells > 1 or face.faces[0] != 4:
                raise ValueError("Only rectangular port faces are supported.")
            
            p1 = np.array(self.pos_to_idx(np.min(face.points, axis=0), mode="edge"))
            p2 = np.array(self.pos_to_idx(np.max(face.points, axis=0), mode="edge"))
    
            # get width of pec in cell units
            idx_d = p2 - p1
    
            if np.count_nonzero(idx_d) != 2:
                raise ValueError("Port face must be on cartesian grid, and must be 2D.")
            
            # normal axis to the port face.
            n_axis = np.argmin(idx_d)
            # axis along integration direction
            f_axis = dict(x=0, y=1, z=2)[integration_axis[0]]
            # axis along width of port
            w_axis = [ax for ax in [0, 1, 2] if ax != f_axis and ax != n_axis][0]
            # direction of integration
            f_dir = {"+": 1, "-": -1}[integration_axis[1]]

            # check that port faces are at least 2 cells wide. The two end points are not included, requiring
            # at least one field component in the center of the face
            if port is not None and idx_d[w_axis] < 2: 
                raise NotImplementedError("Port faces must span at least 2 grid cells.")
            
            # remove endpoints along width if this is a port 
            if port is not None:
                p1[w_axis] += 1
                p2[w_axis] -= 1

            # check that integration axis is perpendicular to normal axis
            if f_axis == n_axis:
                raise ValueError("Integration direction must be in the port face.")

            # string values for each axis, normal, width and field (integration)
            na, wa, fa = [["x", "y", "z"][ax] for ax in [n_axis, w_axis, f_axis]]

            def build_idx(normal, width, field):
                """ Return a tuple of indices into the normal, width and field axis. """
                idx = [slice(None) for i in range(3)]
                idx[w_axis] = width
                idx[f_axis] = field
                idx[n_axis] = normal
                return tuple(idx)
            
            # width of electric field components parallel to the integration axis, a
            dw = dh_cells[w_axis][p1[w_axis] - 1: p2[w_axis]]
            # width of electric field components parallel to the integration axis, along the normal axis
            dn = dh_cells[n_axis][p1[n_axis] - 1: p2[n_axis]]
            # length of field components along the integration axis
            df = d_cells[f_axis][p1[f_axis]: p2[f_axis]]

            # order the width, normal and integration axis so the input to np.meshgrid is xyz, and then broadcast 
            # the cell widths across each other to create a meshgrid the same shape as eps_r.
            # the order of the returned values is x, y, z since they were input in that order by build_index
            dx, dy, dz = np.meshgrid(*build_idx(dn, dw, df), indexing="ij")
            # convert back to dw, dn, df axis notation
            dn, dw, df = [(dx, dy, dz)[ax] for ax in (n_axis, w_axis, f_axis)]

            # epsilon of resistor cells
            eps_field = [self.eps_ex, self.eps_ey, self.eps_ez][f_axis]
            # field indices for the resistive components
            efield_idx = build_idx(
                slice(p1[n_axis], p2[n_axis] + 1), slice(p1[w_axis], p2[w_axis] + 1), slice(p1[f_axis], p2[f_axis])
            )
            eps_r = eps_field[efield_idx]

            # number of components along width. Component is on the edges along the width axis
            nw = (p2 - p1 + 1)[w_axis]
            # number of components along integration axis. Component is in the middle of cell along the integration
            # axis 
            nf = (p2 - p1)[f_axis]
            # resistance of each cell is spilt so the combined resistance of all cells equals r
            r_cell = r * (nw) / nf

            # assign coefficients for resistive element
            rterm = (r_cell * dn * dw)
            denom = (eps_r / self.dt) + (df / (2 * rterm))

            self.Ca[f"e{fa}_{wa}"][efield_idx] = ((eps_r / self.dt) - (df / (2 * rterm))) / denom
            self.Cb[f"e{fa}_{wa}"][efield_idx] = 1 / (denom)
    
            self.Ca[f"e{fa}_{na}"][efield_idx] = ((eps_r / self.dt) - (df / (2 * rterm))) / denom
            self.Cb[f"e{fa}_{na}"][efield_idx] = 1 / (denom)
            
            if port is not None:
                # assign face as a port if specified. resistive element acts as the port termination.
                # direction indicates the voltage integration direction from higher voltage to lower.
                self.ports[port - 1] = dict(
                    idx = efield_idx, 
                    Vs_a = f_dir / (denom * rterm * nf), 
                    field = f"e{fa}", 
                    axis = n_axis, 
                    r0 = r, 
                    direction = f_dir
                )

                # add current probe of current around the termination. current face is normal to the 
                # integration direction
                fpos1 = np.min(face.points, axis=0)
                fpos2 = np.max(face.points, axis=0)
                # position of current surface along integration axis
                field_loc = self.floc[f"e{fa}"][f_axis]
                fpos_n = field_loc[int((p2 + p1)[f_axis] / 2)]
                
                # make the current face slightly larger than lumped element surface so it encloses the current
                current_face = pv.Rectangle([
                    build_idx(fpos1[n_axis] - 0.001, fpos1[w_axis] - 0.001, fpos_n),
                    build_idx(fpos1[n_axis] - 0.001, fpos2[w_axis] + 0.001, fpos_n),
                    build_idx(fpos1[n_axis] + 0.001, fpos2[w_axis] + 0.001, fpos_n),
                ])

                self.add_current_probe(f"port{port}", current_face)


    def _init_PML(self, side):
        """
        Add PML layer to the top face of the solution box.
        """

        n_pml = self.n_pml

        axis = side[0]
        axis_i = dict(x=0, y=1, z=2)[axis]
        m_pml = 3 # sigma profile order

        dt = self.dt
        dcells = self.d_cells[axis_i]
        d_border = conv.m_in(dcells[-1]) if side[1] == "+" else conv.m_in(dcells[0])
        eta0 = np.sqrt(u0 / e0)
        # now define the values of sigma and sigma_m from the profiles
        sigma_max = 0.8 * (m_pml + 1) / (eta0 * d_border)
    
        # define sigma profile in the PML region on the right side of the grid.
        i_pml_axis = np.arange(0, n_pml)
        # broadcast across other dimensions that are not the PML direction
        i_pml_b = [None] * 3
        i_pml_b[axis_i] = slice(None)
        i_pml = i_pml_axis[*i_pml_b]
    
        # sigma on the cell edges. Components on the edge of the PML have a sigma of 0.
        sigma_e_n = sigma_max * ((i_pml) / (n_pml))**m_pml
        # sigma in the middle of the cells. First Hz component in the PML is 0.5 cells into the PML
        sigma_e_np5 = sigma_max * ((i_pml + 0.5) / (n_pml))**m_pml

        # magnetic conductivity
        # plt.figure()
        # plt.plot(np.arange(0, n_pml, 1), sigma_e_n.squeeze())
        # plt.plot(np.arange(0.5, n_pml + .5, 1), sigma_e_np5.squeeze())

        e_idx = [slice(None) for i in range(3)]
        h_idx = [slice(None) for i in range(3)]

        e_idx[axis_i] = slice(n_pml, 0, -1) if side[1] == "-" else slice(-n_pml-1, -1)
        h_idx[axis_i] = slice(n_pml-1, None, -1) if side[1] == "-" else slice(-n_pml, None)

        field_eps = dict(
            ex=self.eps_ex,
            ey=self.eps_ey,
            ez=self.eps_ez,
            hx=self.eps_hx,
            hy=self.eps_hy,
            hz=self.eps_hz,  
        )

        # get the two e and h fields that are graded by the PML for the given axis direction
        pml_efields = [("ey", "ez"), ("ez", "ex"), ("ex", "ey")][axis_i]
        pml_hfields = [("hy", "hz"), ("hz", "hx"), ("hx", "hy")][axis_i]

        # grade the e-field components along axis
        for e in pml_efields:
            # first e component is at the edge of the PML where sigma = 0, last component is at the solve boundary 
            # and not updated.
            # sigma / eps must be constant across y and z, page 291 in taflove
            # scale sigma by eps so that sigma / eps is constant
            eps = field_eps[e][*e_idx]
            sigma_e = np.broadcast_to(sigma_e_n, eps.shape).copy()
            sigma_e *= (eps / e0)
            
            self.Ca[f"{e}_{axis}"][*e_idx] = (2 * eps - (sigma_e * dt)) / (2 * eps + (sigma_e * dt))
            self.Cb[f"{e}_{axis}"][*e_idx] = (2 * dt) / ((2 * eps + (sigma_e * dt)))

        # grade the h-field components along axis
        for h in pml_hfields:
            # h components are in the middle of the PML cells, use half cell indices
            eps = field_eps[h][*h_idx]
            # electrical conductivity
            simga_e = np.broadcast_to(sigma_e_np5, eps.shape).copy()
            simga_e *= (eps / e0)
            # magnetic conductivity
            sigma_m = simga_e * u0 / eps

            self.Da[f"{h}_{axis}"][*h_idx] = (2 * u0 - (sigma_m * dt)) / (2 * u0 + (sigma_m * dt))
            self.Db[f"{h}_{axis}1"][*h_idx] = (2 * dt) / ((2 * u0 + (sigma_m * dt))) 
            self.Db[f"{h}_{axis}2"][*h_idx] = (2 * dt) / ((2 * u0 + (sigma_m * dt))) 


    def gaussian_source(self, width: float, t0: float, t_len: float):
        """
        Generate a gaussian source waveform with the given width in seconds. t_len is the total time of the 
        simulation
        """
        # number of time steps in pulse
        n_len = int(t_len / self.dt)

        t = np.linspace(0, self.dt * n_len, n_len)
        vsrc = np.exp(-((t - t0) / (width / 4)) ** 2) # np.sin(2* np.pi * f0 * t) * 

        # normalize so amplitude is 1
        return (vsrc / np.max(np.abs(vsrc))).astype(np.float32)
    
    def gaussian_modulated_source(self, f0: float, width: float, t0: float, t_len: float):
        """
        Generate a gaussian modulated sinusoidal source waveform with the given width in seconds. 
        t_len is the total time of the simulation
        """
        # number of time steps in pulse
        n_len = int(t_len / self.dt)

        t = np.linspace(0, self.dt * n_len, n_len)
        vsrc = np.exp(-((t - t0) / (width / 4)) ** 2) * np.sin(2* np.pi * f0 * t)

        # normalize so amplitude is 1
        return (vsrc / np.max(np.abs(vsrc))).astype(np.float32)

    def run(self, ports, v_waveforms, n_threads=4, show_progress=True):

        self.check_mesh()

        if isinstance(ports, int):
            ports = [ports]

        v_waveforms = np.atleast_2d(v_waveforms).astype(np.float32)
            
        Nt = len(v_waveforms[0])

        # numpy type for the field values
        dtype_ = np.float32
        dx, dy, dz = [conv.m_in(d).astype(dtype_) for d in self.d_cells]
        dx_h, dy_h, dz_h = [conv.m_in(d).astype(dtype_) for d in self.dh_cells]
        Nx, Ny, Nz = len(dx), len(dy), len(dz)

        dx_inv = 1 / dx[:, None, None]
        dy_inv = 1 / dy[None, :, None]
        dz_inv = 1 / dz[None, None, :]

        dx_h_inv = 1 / dx_h[:, None, None]
        dy_h_inv = 1 / dy_h[None, :, None]
        dz_h_inv = 1 / dz_h[None, None, :]

        # dx with an extra component for the last component at the x edge of the grid that has no neighboring
        # cell.
        dx1_h_inv = np.vstack([dx_h_inv, [[[0]]]])

        # to avoid inefficiently indexing the coefficients at every update, the ends of the coefficients are indexed
        # out so they can be directly multiplied with the difference operations in the update equations. 
        # The grid in the C++ solver is parallelized along x, each x cell is defined as the Ex, Hz, Hy components, and
        # the Ey, Ez, and Hx components at the right end of the cell. The Ey, Ez, and Hx components at the left end of
        # the grid are not included in any cell and are not updated (PEC boundary.) 
        coefficients = dict(
            # ex coefficients, edges along y and z do not get updated
            Ca_ex_y = np.array(self.Ca["ex_y"][:, 1:-1, 1:-1], order="C", dtype=dtype_),
            Ca_ex_z = np.array(self.Ca["ex_z"][:, 1:-1, 1:-1], order="C", dtype=dtype_),
            
            Cb_ex_y = np.array(self.Cb["ex_y"][:, 1:-1, 1:-1] * dy_h_inv, order="C", dtype=dtype_),
            Cb_ex_z = np.array(-self.Cb["ex_z"][:, 1:-1, 1:-1] * dz_h_inv, order="C", dtype=dtype_),

            # ey coefficients, edges along x and z do not get updated
            Ca_ey_z = np.array(self.Ca["ey_z"][1:, :, 1:-1], order="C", dtype=dtype_),
            Ca_ey_x = np.array(self.Ca["ey_x"][1:, :, 1:-1], order="C", dtype=dtype_),
            
            Cb_ey_z = np.array(self.Cb["ey_z"][1:, :, 1:-1] * dz_h_inv, order="C", dtype=dtype_),
            Cb_ey_x = np.array(-self.Cb["ey_x"][1:, :, 1:-1] * dx1_h_inv, order="C", dtype=dtype_),

            # ez coefficients, edges along x and y do not get updated
            Ca_ez_x = np.array(self.Ca["ez_x"][1:, 1:-1, :], order="C", dtype=dtype_),
            Ca_ez_y = np.array(self.Ca["ez_y"][1:, 1:-1, :], order="C", dtype=dtype_),
            
            Cb_ez_x = np.array(self.Cb["ez_x"][1:, 1:-1, :] * dx1_h_inv, order="C", dtype=dtype_),
            Cb_ez_y = np.array(-self.Cb["ez_y"][1:, 1:-1, :] * dy_h_inv, order="C", dtype=dtype_),

            # hx coefficients
            Da_hx_y = self.Da["hx_y"][1:],
            Da_hx_z = self.Da["hx_z"][1:],
            
            Db_hx_y1 = -self.Db["hx_y1"][1:] * dy_inv,
            Db_hx_y2 = -self.Db["hx_y2"][1:] * dy_inv,
            Db_hx_z1 = self.Db["hx_z1"][1:] * dz_inv,
            Db_hx_z2 = self.Db["hx_z2"][1:] * dz_inv,

            # hy coefficients
            Da_hy_z = self.Da["hy_z"],
            Da_hy_x = self.Da["hy_x"],
            
            Db_hy_z1 = -self.Db["hy_z1"] * dz_inv,
            Db_hy_z2 = -self.Db["hy_z2"] * dz_inv,
            Db_hy_x1 = self.Db["hy_x1"] * dx_inv,
            Db_hy_x2 = self.Db["hy_x2"] * dx_inv,

            # hz coefficients
            Da_hz_x = self.Da["hz_x"],
            Da_hz_y = self.Da["hz_y"],
            
            Db_hz_x1 = -self.Db["hz_x1"] * dx_inv,
            Db_hz_x2 = -self.Db["hz_x2"] * dx_inv,
            Db_hz_y1 = self.Db["hz_y1"] * dy_inv,
            Db_hz_y2 = self.Db["hz_y2"] * dy_inv,
        )

        temp_mem_size = 0
        for s in self.fshape.values():
            # three copies of each field, two for the split fields and one combined field
            # the first components at x=0 are not updated and not included in the memory buffer
            temp_mem_size += 3 * np.prod((Nx,) + s[1:])

        # add a buffer for each thread, and the endpoints for the edge components
        temp_mem_size += (4 * (Ny + 1) * (Nz + 1)) * n_threads * 8

        mem = np.zeros(temp_mem_size, dtype=dtype_)

        # initialize probes and sources. Sources act like probes, but the values are input to the 
        # field grid before being replaced by the actual component value.
        probes = []
        for i, p in enumerate(ports):
            port = self.ports[p-1]
            idx, Vs_a, field = port["idx"], port["Vs_a"], port["field"]

            self.ports[p-1]["src"] = v_waveforms[i].copy()

            # convert slice indices to a list of values
            idx_list = [list(np.arange(v.start, v.stop)) if isinstance(v, slice) else [v] for v in idx]

            # create a list of sources for each ez component, with the integer index and scalar waveform data
            Vs_a_flt = Vs_a.flatten()
            for j, idx_j in enumerate(itertools.product(*idx_list)):
                probes.append(
                    dict(
                        values=np.array(Vs_a_flt[j] * v_waveforms[i], dtype=dtype_, order="C"), 
                        field=int(list(self.fshape.keys()).index(field)),
                        idx=[int(id) for id in idx_j],
                        is_source=int(1)
                    )
                )

        # initialize probes
        for k, p in self.probes.items():
            probes.append(
                dict(
                    values=np.zeros(Nt, dtype=dtype_, order="C"), 
                    field=int(list(self.fshape.keys()).index(p["field"])),
                    idx=[int(id) for id in p["index"]],
                    is_source=int(0)
                )
            )

        # initialize field monitors
        monitors = []
        for k, m in self.monitors.items():

            n_m = int(Nt / m["n_step"]) + 1

            monitors.append(
                dict(
                    values=np.zeros(((n_m,) + m["shape"]), dtype=dtype_, order="C"), 
                    axis=int(m["axis"]),
                    position=int(m["index"]),
                    field=list(self.fshape.keys()).index(m["field"]),
                    n_step=int(m["n_step"])
                )
            )

        if show_progress:
            sys.stdout.flush()
            print(f"Running solver with {self.Nx * self.Ny * self.Nz / 1e3:.1f}k cells, and {Nt} time steps...")
            update_interval = int(Nt / 20)
            stime = time.time()
        else:
            update_interval = 0

        core.core_func.solver_run(coefficients, probes, monitors, mem, Nx, Ny, Nz, Nt, n_threads, update_interval)

        if show_progress:
            sys.stdout.write(f"\rDone in {time.time() - stime:.3f}s" + (" " * 20) + "\n")

        # move monitor values back to the class variable
        for i, (k, m) in enumerate(self.monitors.items()):
            self.monitors[k]["values"] = monitors[i]["values"]

        # get the voltages at each source components
        src_v = [s["values"] for s in probes]
        # move the measured source voltages back to the class variable for the associated port
        cur_source = 0
        for p in (ports):
            # get the number of components associated with this port
            src_len = self.ports[p-1]["Vs_a"].size
            src_shape = self.ports[p-1]["Vs_a"].shape

            self.ports[p-1]["values"] = np.array(src_v[cur_source: cur_source + src_len]).reshape(src_shape + (Nt,))
            cur_source += src_len

        # move probe values to the class variable
        for i, (k, p) in enumerate(self.probes.items()):
            self.probes[k]["values"] = probes[i + cur_source]["values"]

        self._solved = True


    def add_field_monitor(self, name: str, field: str, axis: str, position: float, n_step: int):
        """
        Add field monitor along a slice through the 3D volume

        Parameters
        ----------
        name : str
        field : {'ex', 'ey', 'ez', 'e_total', 'hx', 'hy', 'hz'}
        axis : {'x', 'y', 'z'}
        position : float
            position on axis of the slice
        t_step : int, default: 1
            number of time steps between each capture.
        """

        supported_fields = tuple(self.fshape.keys()) + ("e_total",)

        if field not in supported_fields:
            raise ValueError(f"Unsupported field: {field}. Expecting one of: {supported_fields}")
        if axis not in ('x', 'y', 'z'):
            raise ValueError(f"Unsupported axis: {axis}")

        # get the spatial shape of the field slice
        axis_i = dict(x=0, y=1, z=2)[axis]

        # create three seperate monitors for each component if monitor is for the total field
        if field == "e_total":
            field = ("ex", "ey", "ez")
            name = ([name + "_" + f for f in ("x", "y", "z")])
        else:
            field, name = [field], [name]

        for n, f in zip(name, field):
            shape = list(self.fshape[f])
            axis_len = shape.pop(axis_i)

            # convert position along axis to field index
            full_pos = [0] * 3
            full_pos[axis_i] = position
            idx = int(self.field_pos_to_idx(full_pos, f)[axis_i])

            if idx >= (axis_len - 1):
                raise ValueError("Field position out of bounds")

            self.monitors[n] = dict(
                field=f, axis=axis_i, position=position, index=idx, n_step=n_step, shape=tuple(shape)
            )



    def add_current_probe(self, name: str, face: pv.PolyData):
        """
        
        """
        dx_h, dy_h, dz_h = [conv.m_in(d) for d in self.dh_cells]

        if face.n_cells > 1 or face.faces[0] != 4:
            raise ValueError("Only rectangular current faces are supported.")

        # minimum and maximum extents of current face
        pmin, pmax = np.min(face.points, axis=0), np.max(face.points, axis=0)
        # get axis normal to face
        face_size = pmax - pmin
        axis = np.argmin(face_size)

        if np.count_nonzero(face_size) != 2:
            raise ValueError("Port face must be on cartesian grid, and must be 2D.")

        if axis == 0: # current is along x-axis
            # all components have the same x-index
            x0 = self.field_pos_to_idx(pmin, "hy")[0]

            # hz y-indices on the left and right edges of the ampere loop
            hz_y0 = self.field_pos_to_idx(pmin, "hz")[1] - 1
            hz_y1 = self.field_pos_to_idx(pmax, "hz")[1]
            # hy y-indices on top and bottom of the loop
            hy_y = np.arange(hz_y0 + 1, hz_y1 +1)

            # hy z-indices on the top and bottom of the ampere loop
            hy_z0 = self.field_pos_to_idx(pmin, "hy")[2] - 1
            hy_z1 = self.field_pos_to_idx(pmax, "hy")[2]
            # hy z-indces on the left and right of the loop
            hz_z = np.arange(hy_z0 + 1, hy_z1 +1)

            # add left and right probes, save the cell width in meters as the d variable, the sign
            # indicates which direction the component faces in the ampere loop (defined with the current
            # moving in the +x direction)
            i = 0
            for z in (hz_z):
                self.probes[f"{name}_{i}"] = dict(field="hz", index=(x0, hz_y0, z), d=-dz_h[z-1])
                i += 1
                self.probes[f"{name}_{i}"] = dict(field="hz", index=(x0, hz_y1, z), d=dz_h[z-1])
                i += 1

            # add top and bottom probes
            for y in (hy_y):
                self.probes[f"{name}_{i}"] = dict(field="hy", index=(x0, y, hy_z0), d=dy_h[y-1])
                i += 1
                self.probes[f"{name}_{i}"] = dict(field="hy", index=(x0, y, hy_z1), d=-dy_h[y-1])
                i += 1

        elif axis == 1: # current is along the y axis
            # all components have the same y-index
            y0 = self.field_pos_to_idx(pmin, "hz")[1]

            # hx z-indices on the top and bottom of ampere loop
            hx_z0 = self.field_pos_to_idx(pmin, "hx")[2] - 1
            hx_z1 = self.field_pos_to_idx(pmax, "hx")[2]
            # hz z-indices on the left and right of the loop
            hz_z = np.arange(hx_z0 + 1, hx_z1 +1)

            # hz x-indices on the sides of the loop
            hz_x0 = self.field_pos_to_idx(pmin, "hz")[0] - 1
            hz_x1 = self.field_pos_to_idx(pmax, "hz")[0]
            # hx x-indices on the top and bottom of the loop
            hx_x = np.arange(hz_x0 + 1, hz_x1 +1)

            # add left and right probes, save the cell width in meters as the d variable, the sign
            # indicates which direction the component faces in the ampere loop (defined with the current
            # moving in the +y direction)
            i = 0
            for z in (hz_z):
                self.probes[f"{name}_{i}"] = dict(field="hz", index=(hz_x0, y0, z), d=dz_h[z-1])
                i += 1
                self.probes[f"{name}_{i}"] = dict(field="hz", index=(hz_x1, y0, z), d=-dz_h[z-1])
                i += 1

            # add top and bottom probes
            for x in (hx_x):
                self.probes[f"{name}_{i}"] = dict(field="hx", index=(x, y0, hx_z0), d=-dx_h[x-1])
                i += 1
                self.probes[f"{name}_{i}"] = dict(field="hx", index=(x, y0, hx_z1), d=dx_h[x-1])
                i += 1

        elif axis == 2: # current is along z axis
            # all components have the same z-index
            z0 = self.field_pos_to_idx(pmin, "hx")[2]

            # hx y-indices on the top and bottom of the ampere loop 
            hx_y0 = self.field_pos_to_idx(pmin, "hx")[1] - 1
            hx_y1 = self.field_pos_to_idx(pmax, "hx")[1]
            # hy y-indices on the left and right of the loop
            hy_y = np.arange(hx_y0 + 1, hx_y1 +1)

            # hy x-indices on the left and right edges of the ampere loop
            hy_x0 = self.field_pos_to_idx(pmin, "hy")[0] - 1
            hy_x1 = self.field_pos_to_idx(pmax, "hy")[0]
            # hx x-indices on the top and bottom of the loop
            hx_x = np.arange(hy_x0 + 1, hy_x1 +1)

            # add left and right probes
            i = 0
            for y in (hy_y):
                self.probes[f"{name}_{i}"] = dict(field="hy", index=(hy_x0, y, z0), d=-dy_h[y-1])
                i += 1
                self.probes[f"{name}_{i}"] = dict(field="hy", index=(hy_x1, y, z0), d=dy_h[y-1])
                i += 1

            # add top and bottom probes
            for x in (hx_x):
                self.probes[f"{name}_{i}"] = dict(field="hx", index=(x, hx_y0, z0), d=dx_h[x-1])
                i += 1
                self.probes[f"{name}_{i}"] = dict(field="hx", index=(x, hx_y1, z0), d=-dx_h[x-1])
                i += 1

        else:
            raise NotImplementedError("Current face not supported in the given direction yet")
        
        
    def add_line_probe(self, name: str, field: str, line: pv.PolyData):
        """
        
        """
        # get axis that face is constant over (the normal axis)
        axis = np.argmax(np.any(np.diff(line.points, axis=0), axis=0))
        # start and end position of the line, end in inclusive
        line_start, line_end = np.min(line.points, axis=0), np.max(line.points, axis=0)
        # field indices of the line end points
        ijk_start = list(self.field_pos_to_idx(line_start, field))
        ijk_end = list(self.field_pos_to_idx(line_end, field))

        # check if endpoint lands directly on a field component, if it doesn't, the last index shouldn't be
        # included since field_pos_to_idx returns the index on or directly above the point.
        if self.floc[field][axis][ijk_end[axis]] > (line_end[axis] + self._tol):
            ijk_end[axis] -= 1

        # number of probes along line
        n_probes = ijk_end[axis] - ijk_start[axis] + 1

        # break out into indices for each probe, indices are constant if not on the line axis
        ijk_probes = np.broadcast_to(ijk_start, (n_probes, 3)).copy()
        ijk_probes[:, axis] = np.arange(ijk_start[axis], ijk_end[axis] + 1)

        # cell widths at field components
        fcell_w = self.fcell_w[field]

        # direction of line, +1 if oriented along positive axis direction, -1 if along negative direction
        direction = 1 if line_end[axis] > line_start[axis] else -1

        for i, idx in enumerate(ijk_probes):
            # get cell width along the given axis
            cw = [fcell_w[axis][i] for axis, i in enumerate(idx)][axis]
            self.probes[f"{name}_{i}"] = dict(field=field, index=(idx), d=direction * conv.m_in(cw))
        
        
    def render(self, show_probes: bool = False, point_size=15) -> pv.Plotter:
        """
        Plot the model geometry
        """
        self.check_mesh()

        gx, gy, gz = self.g_edges
        gx_h, gy_h, gz_h = self.g_cells
        
        grid = pv.RectilinearGrid(gx, gy, gz)
        
        plotter = pv.Plotter()
        plotter.enable_parallel_projection()
        # add grid
        plotter.add_mesh(grid, style="wireframe", line_width=0.05, color="k", opacity=0.05)

        # add substrates
        for name, (sub) in self.dielectric.items():
            plotter.add_mesh(sub["obj"], **sub["style"])
            
        # add pec
        for name, cond in self.conductor.items():
            plotter.add_mesh(cond["obj"], **cond["style"])

        # add lumped elements
        for name, ele in self.lumped_element.items():
            plotter.add_mesh(ele["obj"], color="pink", opacity=0.5)

        if show_probes:
            points = []
            for k, probe in self.probes.items():
                floc = self.floc[probe["field"]]
                pos = probe["index"]
                # get physical position of probe from the grid index
                points.append([f[p] for f, p in zip(floc, pos)])

            plotter.add_point_labels(
                points,
                list(self.probes.keys()),
                point_color="red" if probe["field"][0] == "h" else "blue",
                point_size=point_size,
                always_visible=True,
                render_points_as_spheres=True,
                fill_shape=False,
                italic=True,
                margin=1
            )

        plotter.show_grid(font_size=9)

        return plotter

    def plot_coefficients(
        self, 
        field, 
        value, 
        axis, 
        position, 
        vmin=None, 
        vmax=None, 
        opacity=1, 
        cmap="jet", 
        point_size=10, 
        normalization=None
    ):
        """
        Plot FDTD coefficients.


        """
        self.check_mesh()

        plotter = self.render()

        # 
        full_pos = [0] * 3
        axis_i = dict(x=0, y=1, z=2)[axis]
        full_pos[axis_i] = position

        idx = [slice(None)] * 3
        idx[axis_i] = self.field_pos_to_idx(full_pos, field[:2])[axis_i]

        if value == "a":
            values = self.Ca[field] if field[0] == "e" else self.Da[field]
        else:
            values = self.Cb[field] if field[0] == "e" else self.Db[field]

        if normalization:
            values = values / normalization
            
        if vmax is None:
            vmax = np.max(values)
        if vmin is None:
            vmin = np.min(values)
        
        floc = self.floc[field[:2]]

        g = [floc[i] if isinstance(s, slice) else floc[i][s: s+1] for i, s in enumerate(idx)]

        fmesh = pv.RectilinearGrid(*g)
        fmesh.point_data['values'] = np.clip(values[*idx], vmin, vmax).flatten(order="F")
        
        plotter.add_mesh(
            fmesh,
            cmap=cmap,
            scalars="values",
            clim=[vmin, vmax],
            show_scalar_bar=False,
            opacity=opacity,
            style="points",
            interpolate_before_map=False,
            render_points_as_spheres=True,
            lighting=False,
            point_size=point_size
        )

        plotter.add_scalar_bar(
            title=f"{field}, {value}\n", vertical=False, label_font_size=11, title_font_size=14
        )
        
        return plotter
    
    def line_probe_values(self, name: str):

        self.check_solution()

        return np.array([p["values"] for k, p in self.probes.items() if k[:len(name)] == name])

    def vi_probe_values(self, name: str):
        """
        Returns time domain current or voltage for the given probe.
        """
        self.check_solution()

        return np.sum([p["values"] * p["d"] for k, p in self.probes.items() if k[:len(name)] == name], axis=0)

    def get_sparameters(self, frequency: np.ndarray, source_port: int = 1, downsample: bool = True):
        """
        Returns a column of the s-parameter matrix excited by a single port.
        """
        self.check_solution()

        nports = len(self.ports)
        nfrequency = len(frequency)

        src_port = self.ports[source_port-1]

        field_idx = src_port["idx"]
        field = src_port["field"]

        # source port applied voltage
        src_applied = src_port["src"]

        # integration axis along field components
        f_axis = dict(x=0, y=1, z=2)[field[1]]
        # width of each field component along integration line, in meters
        cell_w = conv.m_in(self.d_cells[f_axis][field_idx[f_axis]])
        # source port voltage at each component, shape is x, y, z, time
        # broadcast cell widths across other axis (x,y,z,time)
        cell_w_b = [None] * 4
        cell_w_b[f_axis] = slice(None)
        src_component_v = src_port["values"] * cell_w[tuple(cell_w_b)]

        # add voltage along integration axis, remove unitary axis along port face normal
        direction = src_port["direction"]
        src_vp = direction * np.sum(src_component_v, axis=f_axis).squeeze()

        # average voltage across width of port
        if len(src_vp.shape) > 1:
            src_vp = np.mean(src_vp, axis=0)

        # convert to frequency domain
        V_applied = utils.dtft(src_applied, frequency, 1 / self.dt, downsample)
        V_src = utils.dtft(src_vp, frequency, 1 / self.dt, downsample)

        # get the frequency domain current across each port termination
        I_term = np.zeros((nports, nfrequency), dtype=np.complex128)
        for i, p in enumerate(self.ports):

            # voltage generated by termination due the current through it. 
            ip = self.vi_probe_values(f"port{i+1}")

            # h-fields are 1/2 time step ahead of the e-fields. Dely current so they are at the same time step
            I_term[i] = utils.dtft(ip, frequency, 1 / self.dt, downsample) #* np.exp(-1j * frequency * 2 * np.pi * (self.dt / 2))

        # exiting waves (B) from each port
        B = np.zeros((nfrequency, nports), dtype=np.complex128)

        # source port S11, reflected wave (b) is the difference of the total voltage across the port,
        # and the incident wave V = a + b
        B[:, source_port-1] = V_src - V_applied

        # alternate way to separate out foward and reverse waves using the voltage and current.
        # see equation 4.58 in Pozar 4th ed.
        # r_src = self.ports[source_port-1]["r0"]
        # I_src = I_term[source_port-1]
        # I_src = I_src * np.exp(1j * 2 * np.pi * frequency * self.dt /2)
        # B[:, source_port-1]  = (V_src - r_src * I_src) / (2)
        # As = (V_src + r_src * I_src) / (2)

        # the exiting waves on other ports is the voltage that appears across the terminations. This is
        # different than the total voltage across the port because that is the sum of the reflected wave that
        # doesn't make it into the load, and the wave transmitted through the load.
        # The positive current direction for the exiting b wave is into the termination from the line, I_term is 
        # defined as the current along the z axis so the current is inverted. 
        exit_ports = [i for i in range(nports) if i+1 != source_port]
        for i in exit_ports:
            B[:, i] = -I_term[i] * self.ports[i]["r0"]
        
        # return a single column of the full s-matrix
        return B / V_applied[..., None]
    
    def edge_correction(self, p1: tuple, p2: tuple, integration_axis: str, CFe: float = None, CFh: float = 1):
        """
        Analytic correction factors for singularities at PEC surface edges. Corrects asymptotic Ez, Hy fields 
        along the z-axis at the face edge, and Ey, Hz fields along the y-axis at the face edge.

        Parameters
        ----------
        p1 : tuple
            coordinates of first end point of PEC edge to apply the correction to.
        p2 : tuple
            coordinates of second PEC edge end point
        integration_axis : {"x+", "x-", "y-", "y+", "z-", "z+"}
            perpendicular direction pointing away from the edge along which the fields vary asymptotically.
            This should be in the plane of the PEC surface.
        
        """
        self.invalidate_solution()
        self.check_mesh()

        if CFe is None:
            CFe = 2 * np.sqrt(1/2)

        # error check that the line between p1 and p2 is along the x/y/z axis
        p1, p2 = np.array(p1), np.array(p2)
        edge_len = np.abs(p2 - p1)

        if np.count_nonzero(edge_len) != 1:
            raise NotImplementedError("Edge correction is only supported for edges along the cartesian axis.")
        
        # cartesian axis parallel to the edge
        e_axis = np.argmax(edge_len)

        # axis parallel to fields pointing into the PEC edge
        f_axis = dict(x=0, y=1, z=2)[integration_axis[0]]
        # direction pointing away from edge
        f_dir = {"+": 1, "-": 0}[integration_axis[1]]

        # check that integration axis is perpendicular to edge axis
        if e_axis == f_axis:
            raise ValueError("Integration axis must be perpendicular to the edge.")

        # flip points so p2 is at a higher spatial position along the axis
        if p1[e_axis] > p2[e_axis]:
            p2, p1 = p1, p2

        # get e-field component indices at grid edges
        p1_i = self.pos_to_idx(p1, mode="edge")
        p2_i = self.pos_to_idx(p2, mode="edge")

        # axis normal to surface
        n_axis = [ax for ax in [0, 1, 2] if ax != e_axis and ax != f_axis][0]

        # string values for each axis
        ea, fa, na = [["x", "y", "z"][ax] for ax in [e_axis, f_axis, n_axis]]

        def build_idx(edge, field, normal):
            """ Return a tuple of indices into the edge, field and normal axis. """
            idx = [slice(None) for i in range(3)]
            idx[e_axis] = edge
            idx[f_axis] = field
            idx[n_axis] = normal
            return tuple(idx)
        
        # indices on edge axis of the components on cell centers
        e_idx_centers = slice(p1_i[e_axis], p2_i[e_axis])
        # indices on edge axis of the edge components
        e_idx_edges = slice(p1_i[e_axis], p2_i[e_axis] + 1)

        # index of closest H component normal to the surface along field axis
        fh_idx = p1_i[f_axis] if f_dir else p1_i[f_axis] - 1
        # index of surface plane along the normal axis
        n_idx = p1_i[n_axis]

        # assign edge correction coefficents.
        # Comments are for a PEC edge along the x axis, normal to the z-axis, with the field axis along y

        # correct Hz components that integrate the Ey component in the same plane as the PEC.
        # Both Hz and Ey are asymtotic so the correction factor cancels out on all components but the 
        # ex integration.
        # self.Db["hz_y2"][x0: x1, y, z0] *= 1 / CFe
        # self.Db["hz_y1"][x0: x1, y, z0] *= 1 / CFe
        idx = build_idx(e_idx_centers, fh_idx, n_idx)
        self.Db[f"h{na}_{fa}2"][tuple(idx)] *= 1 / CFe
        self.Db[f"h{na}_{fa}1"][tuple(idx)] *= 1 / CFe

        # hz components integrating Ey on the end points of the edge
        if p1_i[e_axis] > 0:
            # self.Db["hz_x2"][x0-1, y, z0] *= CFe
            idx = build_idx(p1_i[e_axis] - 1, fh_idx, n_idx)
            self.Db[f"h{na}_{ea}2"][idx] *= CFe
        if p2_i[e_axis] < self.Db[f"h{na}_{ea}1"].shape[e_axis]:
            # self.Db["hz_x1"][x1, y, z0] *= CFe
            idx = build_idx(p2_i[e_axis], fh_idx, n_idx)
            self.Db[f"h{na}_{ea}1"][idx] *= CFe

        # Correct Hx above and below the PEC plane that integrates Ey 
        # self.Db["hx_z2"][x0: x1+1, y, z0-1] *= CFe
        # self.Db["hx_z1"][x0: x1+1, y, z0] *= CFe
        self.Db[f"h{ea}_{na}2"][build_idx(e_idx_edges, fh_idx, n_idx-1)] *= CFe
        self.Db[f"h{ea}_{na}1"][build_idx(e_idx_edges, fh_idx, n_idx)] *= CFe

        for ni in [n_idx-1, n_idx]:
            # Correct Hx on the sides of the Ez component 
            # self.Db["hx_y2"][x0: x1+1, y0-1, z] *= CFe
            # self.Db["hx_y1"][x0: x1+1, y0, z] *= CFe
            self.Db[f"h{ea}_{fa}2"][build_idx(e_idx_edges, p1_i[f_axis] -1, ni)] *= CFe
            self.Db[f"h{ea}_{fa}1"][build_idx(e_idx_edges, p1_i[f_axis], ni)] *= CFe

            # correct Hy components that integrate the Ez component below and above the edge 
            # self.Db["hy_z1"][x0: x1, y0, z] *= 1 / CFe
            # self.Db["hy_z2"][x0: x1, y0, z] *= 1 / CFe
            self.Db[f"h{fa}_{na}1"][build_idx(e_idx_centers, p1_i[f_axis], ni)] *= 1 / CFe
            self.Db[f"h{fa}_{na}2"][build_idx(e_idx_centers, p1_i[f_axis], ni)] *= 1 / CFe

            # hy components integrating Ez on the end points of the edge
            if p1_i[e_axis] > 0:
                # self.Db["hy_x2"][x0-1, y0, z] *= CFe
                idx = build_idx(p1_i[e_axis] - 1, p1_i[f_axis], ni) 
                self.Db[f"h{fa}_{ea}2"][idx] *= CFe
            if p2_i[e_axis] < self.Db[f"h{fa}_{ea}1"].shape[e_axis]:
                # self.Db["hy_x1"][x1, y0, z] *= CFe
                idx = build_idx(p2_i[e_axis], p1_i[f_axis], ni)
                self.Db[f"h{fa}_{ea}1"][idx] *= CFe

        # correct Hz components that use the Ey component in the same plane as the edge that points into the edge.
        # Hz in the same plane as the face. Both Hz and Ey are asymtotic so the correction factor cancels out
        # on all components but the ex integration.
        # for y in [y0-1, y1]:
        #     self.Db["hz_y2"][x0: x1, y, z0] *= 1 / CF
        #     self.Db["hz_y1"][x0: x1, y, z0] *= 1 / CF
        #     # hz components using Ey that are just past the face along the x direction
        #     self.Db["hz_x2"][x0-1, y, z0] *= CF
        #     if not x1_at_edge:
        #         self.Db["hz_x1"][x1, y, z0] *= CF
        #     # Hx just below the Ey component
        #     self.Db["hx_z2"][x0: x1+1, y, z0-1] *= CF
        #     # Hx just above the Ey component
        #     self.Db["hx_z1"][x0: x1+1, y, z0] *= CF

        # # correct Hy components that use the Ez component below and above the edge 
        # # Hy and Ez are both asymptotic and the correction factors cancel out
        # for y in [y0, y1]:
        #     self.Db["hy_z1"][x0: x1, y, z0-1] *= 1 / CF
        #     self.Db["hy_z2"][x0: x1, y, z0-1] *= 1 / CF
        #     self.Db["hy_z1"][x0: x1, y, z0] *= 1 / CF
        #     self.Db["hy_z2"][x0: x1, y, z0] *= 1 / CF
        #     # hy components just past the face along the x direction
        #     self.Db["hy_x2"][x0-1, y, z0-1] *= CF
        #     self.Db["hy_x2"][x0-1, y, z0] *= CF
        #     if not x1_at_edge:
        #         self.Db["hy_x1"][x1, y, z0] *= CF
        #         self.Db["hy_x1"][x1, y, z0-1] *= CF

        # # correct Hx components just past the face in the y direction that use the Ez components under the edge
        # for z in [z0-1, z0]:
        #     self.Db["hx_y2"][x0: x1+1, y0-1, z] *= CF
        #     self.Db["hx_y1"][x0: x1+1, y0, z] *= CF
        #     self.Db["hx_y2"][x0: x1+1, y1-1, z] *= CF
        #     self.Db["hx_y1"][x0: x1+1, y1, z] *= CF

        # correction E components whose integration plane is a half a cell away from the edge, Ez, and Ey
        # if CFh is not None:
        #     for z in [z0-1, z0]:
        #         self.Cb["ez_y"][x0+1: x1, y0, z] *= 1 / CFh
        #         self.Cb["ez_y"][x0+1: x1, y1, z] *= 1 / CFh

        #     for y in [y0-1, y1]:
        #         self.Cb["ey_z"][x0+1: x1, y, z0] *= 1 / CFh

    
    def get_monitor_data(self, name: str) -> ldarray:
        """
        Compile field monitor data.

        Parameters
        ----------
        name : str
            name of monitor

        Returns
        -------
        ldarray
            labeled numpy array with time dimension and two spatial dimensions along monitor plane.
            If monitor is a total field monitor, the first dimension is the three rectangular components.
        """
        self.check_solution()

        # component field names
        c_names = [f"{name}_{a}" for a in ("x", "y", "z")]
        # is there monitor data for all three rectangular components
        is_total_field = all([n in self.monitors.keys() for n in c_names])

        monitor = self.monitors[c_names[0]] if is_total_field else self.monitors[name]
        # get the two component axis on the monitor plane
        spatial_axis = [0, 1, 2]
        spatial_dims = ["x", "y", "z"]
        spatial_axis.pop(monitor["axis"])

        # axis normal to monitor plane
        axis = monitor["axis"]
        axis_str = ["x", "y", "z"][axis]

        # time vector
        t_len = len(monitor["values"]) * self.dt * monitor["n_step"]
        time_values = np.arange(0, t_len, self.dt * monitor["n_step"], dtype=np.float64)
        
        # combine component vectors into a single matrix
        if is_total_field:
            e_xyz = [self.monitors[n]["values"] for n in c_names]
            # order the fields by the components in the surface, followed by the normal component.
            # a surface on xy will be ordered x, y, z. xz will be ordered x, z, y., yz will be ordered, y, z, x.
            axis_surf_ordered = spatial_axis + [axis]
            e1, e2, e3 = [e_xyz[a] for a in axis_surf_ordered]

            # average components on the xy plane to get the fields at the cell corners.
            # fields at the grid edge are dropped
            e1 = (e1[:, 1:, 1:-1] + e1[:, :-1, 1:-1]) / 2
            e2 = (e2[:, 1:-1, 1:] + e2[:, 1:-1, :-1]) / 2
            e3 = (e3[:, 1:-1, 1:-1])

            # order back to x, y, z
            e_xyz = [(e1, e2, e3)[a] for a in axis_surf_ordered]

            # if axis_str == "z":
            #     ex = (ex[:, 1:, 1:-1] + ex[:, :-1, 1:-1]) / 2
            #     ey = (ey[:, 1:-1, 1:] + ey[:, 1:-1, :-1]) / 2
            #     ez = (ez[:, 1:-1, 1:-1])
            # # average components on the xz plane to get the fields on the corners
            # elif axis_str == "y":
            #     ex = (ex[:, 1:, 1:-1] + ex[:, :-1, 1:-1]) / 2
            #     ey = (ey[:, 1:-1, 1:-1])
            #     ez = (ez[:, 1:-1, 1:] + ez[:, 1:-1, :-1]) / 2
            # # average components on the yz plane
            # else: # axis == "x":
            #     ex = (ex[:, 1:-1, 1:-1])
            #     ey = (ex[:, 1:, 1:-1] + ex[:, :-1, 1:-1]) / 2
            #     ez = (ez[:, 1:-1, 1:] + ez[:, 1:-1, :-1]) / 2

            # Assign the spatial coordinates of the grid corners.
            # Use the coordinates from the component normal to the monitor surface since it lies on the corners.
            spatial_coords = {spatial_dims[i]: self.floc[f"e{axis_str}"][i][1:-1] for i in spatial_axis}

            return ldarray(
                e_xyz, 
                coords=dict(component=("x", "y", "z"), time=time_values, **spatial_coords)
            )

        # compile a single component field
        else:
            field = monitor["field"]
            # build coordinates in inches for the two spatial dimensions of the slice
            spatial_coords = {spatial_dims[i]: self.floc[field][i] for i in spatial_axis}

            return ldarray(
                monitor["values"], 
                coords=dict(time=time_values, **spatial_coords)
            )

 
    def plot_monitor(
        self,
        monitor: list,
        linear: bool = False, 
        cmap: str = "jet",
        style: str = "points",
        opacity: str = "linear",
        init_time: float = None,
        max_vector_len: float = 0.01,
        camera_position: str = "xy",
        vmin: float = None,
        vmax: float = None, 
        zoom: float = 1.0, 
        show_rulers: bool = True,
        colorbar_title: str = None,
        plotter: pv.Plotter = None,
        gif_setup: dict = None,
    ) -> pv.Plotter:
        """
        Show a field monitor overlayed on the model geometry. Time step of the fields is controlled with an interactive
        slider bar.

        Parameters
        ----------
        monitor : str | list
            monitor name(s) assigned with add_field_monitor. 

        linear : bool, default: False
            if True, linear field magnitudes are plotted on the overlay. If False, values are converted to db20
            before plotting.

        cmap : str, default: "jet"
            matplotlib colormap name

        style : {"surface", "points", "vectors"}, default: "surface"
            Visualization style of the overlay. If "vectors", monitor must be a total field monitor with three
            component vectors. Can be a list if multiple monitors are given. Values are interpolated for the "surface"
            style and shows a continuous overlay. Values are shown as non-interpolated, discrete points on the 
            Yee grid if "points" is chosen.
            
        opacity : str | list, default: linear
            Opacity of the field overlay. A float value from 0-1 can be provided, or a string can be specified to 
            map the scalars range to a predefined opacity transfer function (options include: 'linear', 
            'linear_r', 'geom', 'geom_r'). If multiple monitors are given, this can be a list to specify different
            opacities to each overlay.

        init_time : float, optional
            initial time step to use for the field overlays. Default is half of the simulation interval.

        max_vector_len : float, default: 0.01
            maximum length of vectors in inches, should be chosen so the vectors are smaller than the average grid
            cell in the overlay to avoid overlapping. Ignored if style is not "surface".
        
        camera_position : str | pv.CameraPosition, default: "xy"
            camera position, passed to pv.Plotter.camera_position
        
        vmin : float, optional
            minimum field value shown on the colormap, must be in dB if linear is false.
        
        vmax : float, optional
            maximum field value shown on the colormap, must be in dB if linear is false.
        
        zoom : float, default: 1
            camera zoom

        show_rulers : bool, default: True
            show the model bounding box with rulers
        
        colorbar_title : str, optional
            title above the colorbar

        plotter : pv.Plotter, optional
            pyvista Plotter object. If provided the model geometry is not drawn on the plot and needs to be drawn
            manually with render().

        gif_setup : dict, optional
            Configuration settings for .gif generation. The required key/value pair is "file", the others are optional.
            file: file path for .gif file (must end in .gif)
            fps: frame per second of gif, default: 15
            loop: number of times to loop gif, default: 0 (infinite loop)
            start_ps: starting time of simulation to include in gif, in picoseconds. Default: 0
            end_ps: ending time of simulation to include in gif, in picoseconds. Default: end of simulation
            step_ps = number of time steps to skip between in each frame, in picoseconds. Default: 1

        Returns
        -------
        pv.Plotter
            pyvista Plotter object. Plotter is closed and cannot be reopened if gif_setup is provided.
        
        """
        self.check_solution()

        # start with the rendered view of the model 
        if plotter is None:
            plotter = self.render()

        monitor = np.atleast_1d(monitor)
        opacity = np.atleast_1d(opacity)
        style = np.atleast_1d(style)

        # time step size of monitors
        n_step = None
        # inital time step frame to start the feild overlays with
        init_frame = None
        # number of total frames
        n_frames = None
        
        # keep track of field values and plot actors so they can be updated interactively
        plot_items = [dict() for i in range(len(monitor))]

        for i, name in enumerate(monitor):

            # component field names
            c_names = [f"{name}_{a}" for a in ("x", "y", "z")]
            # is there monitor data for all three rectangular components
            is_total_field = all([n in self.monitors.keys() for n in c_names])

            # if total field monitor, use the monitor of the first component to get the axis and axis position
            monitor = self.monitors[c_names[0]] if is_total_field else self.monitors[name]
            axis = monitor["axis"]
            axis_pos = monitor["position"]

            # initialize time step values if on the first monitor
            if n_step is None:
                n_step = monitor["n_step"]
                # number of frames in the monitor data, time dimension always preceds the 2 spatial dims of the surface
                n_frames = monitor["values"].shape[-3]
                # frame index to the monitor data
                init_frame = int(init_time / n_step) if init_time is not None else int(n_frames / 2)

            # check that all monitors have the same time step size
            elif monitor["n_step"] != n_step:
                raise ValueError("All monitors must have identical time steps.")

            field = self.get_monitor_data(name)

            # get the two component axis on the monitor plane
            surf_axis = [0, 1, 2]
            surf_axis.pop(monitor["axis"])
            axis_ordered = surf_axis + [axis]

            # meshgrid of coordinates on the surface
            surf_coords_m = np.meshgrid(*list(field.coords.values())[-2:], indexing="ij")
            # meshgrid of a single value of the axis position of the surface
            n_coords_m = np.ones(surf_coords_m[0].size) * axis_pos
            # all coordinate points, flattened into a the rows, shape is number of points, cartesian xyz
            coords_xyz = [(*surf_coords_m, n_coords_m)[a] for a in axis_ordered]
            points = np.vstack([a.ravel() for a in coords_xyz]).T
            
            # get magnitude if all component vectors are present, and flatten spatial point dimensions.
            # field_v has shape time, flattened spatial points
            if is_total_field:
                field_v = np.sqrt(np.sum(np.abs(field)**2, axis=0)).reshape(len(field[0]), -1)
            else:
                field_v = field.reshape(len(field), -1, order="F")

            # convert to db
            if not linear:
                # avoid nan values for zero values
                field_v = np.where(np.abs(field_v) < 1e-12, 1e-12, field_v)
                field_v = conv.db20_lin(field_v)

            # set min/max bounds automatically if not given
            if vmax is None:
                vmax = np.nanmax(field_v)
            if vmin is None:
                vmin = -vmax if linear else vmax - 50

            # initialize vector field
            if style[i] == "vector":

                if not is_total_field:
                    raise ValueError("Monitor must contain all field components to plot a vector field.")

                # flatten spatial dimensions to create a list of vectors
                vectors = np.reshape(field, (*field.shape[0:2], -1 ))
                # move xyz axis to the end, shape is time, spatial points, cartesian xyz
                vectors = np.transpose(vectors, (1, 2, 0))

                mesh = pv.PolyData(points)
                # assign colormap values
                mesh.point_data['values'] = field_v[init_frame]
                # assign vector direction
                mesh["vectors"] = vectors[init_frame]
                # assign vector lengths. Scale so that minimum value gives a length of 0.
                vector_len = np.clip(field_v, vmin, vmax) - vmin
                vector_len = max_vector_len * (vector_len / np.max(vector_len))
                mesh["mag"] = vector_len[init_frame]

                # create vector field, setting the arrow scaling with the "mag" dataset and direction with the "vector" 
                # dataset
                arrows = mesh.glyph(orient='vectors', scale='mag')
                arrows.set_active_scalars("values")

                actor = plotter.add_mesh(
                    arrows, 
                    scalars="values", 
                    cmap="jet",
                    clim=[vmin, vmax], 
                    show_scalar_bar=False, 
                    opacity=opacity[i]
                )

                plot_items[i] = dict(
                    actor=actor,
                    mesh=mesh,
                    scalars=field_v,
                    arrows=arrows,
                    vectors=vectors,
                    mag=vector_len,
                )

            # initialize colormapped scalar surface
            else:
                # get grid points along each axis
                if axis == 2: # xy plane
                    floc_in = (field.coords["x"], field.coords["y"], [monitor["position"]])
                elif axis == 1: # xz plane
                    floc_in = (field.coords["x"], monitor["position"], field.coords["z"])
                else: # yz plane
                    floc_in = (monitor["position"], field.coords["y"], field.coords["z"])
                
                # create rectangular grid surface and assign scalar field values as the color
                mesh = pv.RectilinearGrid(*floc_in)
                mesh.point_data['values'] = field_v[init_frame]
                
                actor = plotter.add_mesh(
                    mesh, 
                    cmap=cmap, 
                    scalars="values", 
                    clim=[vmin, vmax], 
                    show_scalar_bar=False, 
                    interpolate_before_map=bool(style == "surface"),
                    render_points_as_spheres=True,
                    style=style[i],
                    opacity=opacity[i],
                    lighting=False,
                    point_size=10
                )

                plot_items[i] = dict(
                    actor=actor,
                    mesh=mesh,
                    scalars=field_v,
                )

        # number of time points in the simulation
        Nt = n_frames * n_step
        # round starting point to nearest ps
        self.slider_value = int(init_frame * n_step * self.dt * 1e12)
        self.slider = None

        plotter.add_axes()
        plotter.camera_position = camera_position
        plotter.camera.zoom(zoom)

        # remove rulers that are on by default
        if not show_rulers:
            plotter.remove_bounds_axes()
        
        # add title over colorbar
        if colorbar_title is None:
            colorbar_title="E [V/m]\n" if linear else "E [dB]\n"

        plotter.add_scalar_bar(
            colorbar_title, vertical=False, label_font_size=11, title_font_size=14
        )
        
        def update_fields(time_ps):
            """ function called every time the slider bar is moved. """

            self.slider_value = time_ps
            # time step in simulation
            n = time_ps * 1e-12 / self.dt
            # frame index
            frame = int(n / n_step)
            # skip update if frame is out of bounds
            if frame >= n_frames:
                return
            
            # force update slider value. This allows the slider to be set programmatically 
            if self.slider is not None:
                self.slider.GetRepresentation().SetValue(self.slider_value)

            # iterate through monitor plot items
            for p in plot_items:
                # update scalar colormap values
                mesh = p["mesh"]
                mesh.point_data["values"][:] = p["scalars"][frame]

                # update vector direction and magnitude
                if "arrows" in p.keys():
                    
                    for value in ("vectors", "mag"):
                        mesh[value][:] = p[value][frame]

                    p["arrows"].copy_from(mesh.glyph(orient='vectors', scale='mag'))
                    p["arrows"].set_active_scalars("values")

        # add slider widget
        self.slider = plotter.add_slider_widget(
            update_fields,
            [0, Nt * self.dt * 1e12],
            value=self.slider_value,
            title="Time [ps]",
            interaction_event="always",
            style="modern",
            fmt="%0.2f"
        )

        # generate gif
        if gif_setup is not None:
            file_ = gif_setup["file"]
            fps = gif_setup.get("fps", 15)
            loop = gif_setup.get("loop", 0)

            # start and end time in ps of gif
            start_ps = gif_setup.get("start_ps", 0)
            end_ps = gif_setup.get("end_ps", Nt * self.dt * 1e12)
            step_ps = gif_setup.get("step_ps", 1)

            plotter.open_gif(file_, fps=fps, loop=loop, subrectangles=True)

            # step through each frame in the monitor, skipping by frame_step on each iteration
            for time_ps in np.arange(start_ps, end_ps + step_ps, step_ps):

                # update field overlays
                update_fields(time_ps)
                plotter.write_frame()

            # Closes and finalizes movie
            plotter.close()
            return plotter

        # add checkbox to turn off field visibility
        def set_field_visible(value):
            for p in plot_items:
                p["actor"].SetVisibility(value)

        plotter.add_checkbox_button_widget(set_field_visible, value=True, position=(10, 10), size=30, border_size=0)
        plotter.add_text("Field Visibility", position=(45, 15), font_size=9)

        # add key bindings to increment the slider left/right with the arrow keys
        def increment_left():
            # increment down to nearest picosecond
            update_fields(int(self.slider_value - 1))
            plotter.render()

        def increment_right():
            # increment up to nearest picosecond
            update_fields(int(self.slider_value + 1))
            plotter.render()

        plotter.add_key_event("Left", increment_left)
        plotter.add_key_event("Right", increment_right)
        
        return plotter
    


