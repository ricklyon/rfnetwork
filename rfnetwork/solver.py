import itertools
import time
import numpy as np 

import pyvista as pv

from rfnetwork import const, conv, utils, core

u0 = const.u0
e0 = const.e0
c0 = const.c0


class Solver_PCB():
    """
    3D EM Solver for PCB geometries, substrates are normal to the z-axis.
    """

    def __init__(self, bounding_box, nports=1):
        self.substrate = dict()
        self.pec_face = dict()
        self.styles = dict()
        self.port_face = [None] * nports
        self.bounding_box = bounding_box
        self.monitors=dict()
        self.probes=dict()
        self.slider_value = 0

        self.sbox_max = np.max(bounding_box.points, axis=0)
        self.sbox_min = np.min(bounding_box.points, axis=0)

        self._places = 5 # hundredth of a mil
        self._tol = 1 / (10 ** self._places)

    def add_substrate(self, name, obj, er: float, **kwargs):
        self.substrate[name] = (obj, er)
        self.styles[name] = kwargs

    def add_pec_face(self, name, obj, **kwargs):
        self.pec_face[name] = obj
        self.styles[name] = kwargs

    def add_lumped_port(self, number, face):
        self.port_face[number - 1] = face

    def pos_to_idx(self, p, mode="edge"):
        """
        Returns the index of the grid edge or cell center that is directly on or just past (in +x, +y and +z directions) 
        the given position.
        """
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
        floc = self.floc[field]

        idx = []
        for i, g in enumerate(floc):
            diff = (g - position[i])
            # if no cell is above the point, return the length of the axis, otherwise return the first cell that is 
            # larger than the point.
            idx += [np.argmax(diff >= -self._tol) if diff[-1] > -self._tol else len(g)]
        
        return tuple(idx)
    
    def init_mesh(self, d0, n0, d_pec, n_min_pec, d_sub, n_min_sub, blend_pec = False):
        """
        Initialize the spatial grid.
        """
        edges = [np.array([], dtype=np.float32) for i in range(3)]

        objects = [self.bounding_box] + list(self.pec_face.values()) + [sub[0] for sub in self.substrate.values()]
        dtype_ = np.float32

        # build list of edge coordinates along each axis
        for obj in objects:
            # round points to minimum precision supported by the mesh
            p_edges = np.around(obj.points.T, decimals=self._places).astype(np.float32)
        
            for i in range(3):
                edges_i = np.unique(np.concatenate([edges[i], p_edges[i]]))
                edges[i] = edges_i

        # get bounds of all pec objects
        pec_bounds = []
        for pec in self.pec_face.values():
            pmin, pmax = np.min(pec.points, axis=0), np.max(pec.points, axis=0)
            pec_bounds.append(np.array([pmin, pmax]))

        # rearrange pec bounds so the axis is first, then the pec object, then min/max bounds
        pec_bounds = np.array(pec_bounds).transpose(2, 0, 1)

        # maximum z coordinate of any substrate
        sub_z_max = 0
        # maximum er of any substrate
        sub_er_max = 1
        for sub, sub_er in self.substrate.values():
            sub_z_max = np.max([sub_z_max, np.max(sub.points[:, 2])])
            sub_er_max = np.max([sub_er_max, sub_er])

        # build a list of cell widths along each axis
        cell_d = [[], [], []]
        for axis in range(3):

            # list of distances between each edge
            d_axis_const = np.diff(edges[axis])
            d_axis = []

            dmax_axis = []  # maximum sub-cell size within each cell
            nmin_axis = []  # minimum number of sub-cells within each cell
            is_pec_axis = []

            for i, cell_w in enumerate(d_axis_const):
                # edges of this cell
                cell_min, cell_max = edges[axis][i:i+2]
                # if the cell is inside a PEC object, use PEC mesh settings
                is_pec = False
                if np.any((cell_min + self._tol >= pec_bounds[axis][:, 0]) & (cell_max - self._tol <= pec_bounds[axis][:, 1])):
                    # divide cell into sub-cells that are no larger than d_pec, and at least n_min_pec cells
                    n_min = n_min_pec
                    d_max = d_pec
                    is_pec = True
                # if on the z-axis and cell is inside a substrate, use substrate mesh settings
                elif axis == 2 and (cell_max - self._tol) <= sub_z_max:
                    n_min = n_min_sub
                    d_max = d_sub
                # otherwise use global settings
                else:
                    n_min = n0
                    d_max = d0 

                # split cells larger than d0*5 into two cells, this prevents large cells bounded by 
                # equal sized small ones from being broken up into small cells, and instead ensures
                # the span is graded from small widths to large, and then back down to small widths.
                n_split = 1
                split_threshold = d0 * 5
                if cell_w > split_threshold:
                    n_split = int(cell_w / split_threshold)
                    # split cell width into equal parts
                    cell_w = cell_w / n_split
                    
                # estimate the minimum number of sub-cells this cell must be broken into so no sub-cell is larger
                # than d_max
                cell_sub_n = np.clip(int(cell_w / d_max), n_min, None)
                # maximum distance of any sub-cell within this cell
                dmax_axis += [cell_w / cell_sub_n] * n_split
                # minimum number of sub-cells for this cell that is enforced by mesh settings
                nmin_axis += [n_min] * n_split
                # update the cell width vector
                d_axis += [cell_w] * n_split
                is_pec_axis += [is_pec] * n_split

            # cells indices arranged from largest width to smallest
            # blend large cells first so they transition gradually into smaller ones.
            # The larger cells have more room to work in and
            # it's easier to blend. The smaller cells have less work to do because the
            # larger cells have already stepped down to meet their widths.
            # Leave PEC cells for last so they don't blend up to the (typically) larger d0 cells
            d_order = np.flip(np.argsort(d_axis))

            if not blend_pec:
                # don't blend in PEC cells, put these first in the blending order so non-PEC surrounding cells 
                # are forced to match to the minimum PEC cell width instead of the maximum
                n_pec = np.count_nonzero(is_pec_axis)
                pec_ordered = np.flip(np.argsort(is_pec_axis))[:n_pec]
                # remove pec indices to get list of non-pec indices in d_axis, put pec first in d_order
                indices_to_remove = [i for i, d in enumerate(d_order) if is_pec_axis[d]]
                non_pec_ordered = np.delete(d_order, indices_to_remove)
                d_order = np.concatenate([pec_ordered, non_pec_ordered])

            d_sides = [(d, d) for d in dmax_axis] # sub-cell widths on left and right edge of the cells
            
            d_subcells = [None] * len(d_axis)
            for i in d_order:
                # previous and next cell width
                dprev = d_sides[i - 1][1] if i > 0 else dmax_axis[i]
                dnext = d_sides[i + 1][0] if i < len(d_axis) - 1 else dmax_axis[i]

                # if PEC cell, use constant cell widths matched to the smallest adjacent cell if not using blend pec
                if is_pec_axis[i] and not blend_pec:
                    min_neighbor_w = min(dprev, dnext)
                    # minimum number of sub-cells this cell must be broken up into
                    n = int(np.clip(d_axis[i] / dmax_axis[i], nmin_axis[i], None))

                    # match cell size to the smallest neighboring cell if it meets the min number of sub-cells needed
                    # for this cell
                    if min_neighbor_w < (d_axis[i] / n):
                        rmd = (d_axis[i] % min_neighbor_w)
                        n_subcell = int(d_axis[i] // min_neighbor_w)
                        sub_cell_w = np.array([min_neighbor_w + (rmd / n_subcell)] * n_subcell, dtype=dtype_)
                    # otherwise, break up into the number of required cells
                    else:
                        sub_cell_w = np.array([d_axis[i] / n] * n)
                else:
                    sub_cell_w = list(
                        utils.blend_cell_widths(dprev, dnext, d_axis[i], n_min = nmin_axis[i], tol=self._tol)
                    )

                d_subcells[i] = sub_cell_w
                d_sides[i] = (sub_cell_w[0], sub_cell_w[-1])

            # flatten list of lists of subcell widths
            cell_d[axis] = list(itertools.chain(*d_subcells))

        
        gx, gy, gz = [np.around(np.concatenate([[self.sbox_min[i]], self.sbox_min[i] + np.cumsum(cell_d[i])]), decimals=self._places) for i in range(3)]
        dx, dy, dz = np.diff(gx).astype(dtype_), np.diff(gy).astype(dtype_), np.diff(gz).astype(dtype_)

        self.n_cells = len(dx), len(dy), len(dz)  
        self.grid_mesh = pv.RectilinearGrid(gx.astype(dtype_), gy.astype(dtype_), gz.astype(dtype_))

        self.Nx, self.Ny, self.Nz = self.n_cells
        self.dx, self.dy, self.dz = dx, dy, dz
        
        # locations of cell center
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

        for (sub, er) in self.substrate.values():
            x0, y0, z0 = self.pos_to_idx(np.min(sub.points, axis=0), mode="cell")
            x1, y1, z1 = self.pos_to_idx(np.max(sub.points, axis=0), mode="cell")
        
            self.eps[x0: x1, y0: y1, z0: z1] = e0 * er

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
        


    def init_coefficients(self):
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

        # substrate
        # ez components are on the edge of the x/y cell boundaries, average epsilon from the adjacent cells
        dx0, dx1 = dx[:-1][..., None, None], dx[1:][..., None, None]
        dy0, dy1 = dy[:-1][None, :, None], dy[1:][None, :, None]
        dz0, dz1 = dz[:-1][None, None], dz[1:][None, None]

        eps = self.eps
        # average epsilon cells adjacent to x edges
        eps_x = (eps[:-1] * (dx0/2) + eps[1:] * (dx1/2)) / (dx0/2 + dx1/2)
        # average epsilon cells adjacent to y edges
        eps_y = (eps[:, :-1] * (dy0/2) + eps[:, 1:] * (dy1/2)) / (dy0/2 + dy1/2)
        # average epsilon cells adjacent to z edges
        eps_z = (eps[..., :-1] * (dz0/2) + eps[..., 1:] * (dz1/2)) / (dz0/2 + dz1/2)

        # average epsilon cells along y and z
        eps_yz = ((eps_y[..., :-1] * (dz0/2) + eps_y[..., 1:] * (dz1/2)) / (dz0/2 + dz1/2))
        # average epsilon cells along x and z
        eps_xz = ((eps_x[..., :-1] * (dz0/2) + eps_x[..., 1:] * (dz1/2)) / (dz0/2 + dz1/2))
        # average epsilon cells along x and y
        eps_xy = ((eps_x[:, :-1] * (dy0/2) + eps_x[:, 1:] * (dy1/2)) / (dy0/2 + dy1/2))


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

        # coefficient in front of the previous time values of E
        Ca_ex = np.ones((Nx, Ny+1, Nz+1), dtype=dtype_) * Ca_0
        Ca_ey = np.ones((Nx+1, Ny, Nz+1), dtype=dtype_) * Ca_0
        Ca_ez = np.ones((Nx+1, Ny+1, Nz), dtype=dtype_) * Ca_0
        
        # coefficient in front of the difference terms of H
        Cb_ex = dt / self.eps_ex
        Cb_ey = dt / self.eps_ey
        Cb_ez = dt / self.eps_ez
    
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

        # self.eps_ex_z = self.eps_ex.copy()
        # self.eps_ex_z[..., 4] = e0 * 1
        # self.Cb["ex_z"] = dt / self.eps_ex_z

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

    def init_pec(self, edge_correction=True):
        # initialize the PEC faces
        # this should be called after setting the PML layers

        dx, dy, dz = [conv.m_in(d) for d in self.d_cells]

        dx_h, dy_h, dz_h = [conv.m_in(d) for d in self.dh_cells]

        # PEC pattern
        for name, pec in self.pec_face.items():
            
            if pec.n_cells > 1 or pec.faces[0] != 4:
                raise ValueError("Only rectangular PEC faces are supported.")
            
            pmin, pmax = np.min(pec.points, axis=0), np.max(pec.points, axis=0)
            # normal axis
            axis = np.argmin(pmax - pmin)
    
            if np.count_nonzero(pmax - pmin) != 2:
                raise ValueError("PEC face must be on cartesian grid, and must be 2D.")
                    

            # PEC is normal to the z-axis, Ex and Ey are parallel to surface
            if (axis == 2):
                
                sig_0 = 1e6
                Ca_0 = (2 * e0 - (sig_0 * self.dt)) / (2 * e0 + (sig_0 * self.dt))
                Cb_0 = (2 * self.dt) / ((2 * e0 + (sig_0 * self.dt)))

                # field pos returns the first index that is greater than the given position, so y1 index is just
                # past the edge of the end of the PEC
                x0, y0, z0 = self.field_pos_to_idx(np.min(pec.points, axis=0), "ey")
                x1, y1, z1 = self.field_pos_to_idx(np.max(pec.points, axis=0), "ey")
                self.Cb["ey_z"][x0: x1+1, y0: y1, z0] = 0
                self.Ca["ey_z"][x0: x1+1, y0: y1, z0] = -1
                self.Cb["ey_x"][x0: x1+1, y0: y1, z0] = 0
                self.Ca["ey_x"][x0: x1+1, y0: y1, z0] = -1
                # ex cells on x axis are not inclusive, edges on y axis are
                x0, y0, z0 = self.field_pos_to_idx(np.min(pec.points, axis=0), "ex")
                x1, y1, z1 = self.field_pos_to_idx(np.max(pec.points, axis=0), "ex")
                self.Cb["ex_y"][x0: x1, y0: y1+1, z0] = 0
                self.Ca["ex_y"][x0: x1, y0: y1+1, z0] = -1
                self.Cb["ex_z"][x0: x1, y0: y1+1, z0] = 0
                self.Ca["ex_z"][x0: x1, y0: y1+1, z0] = -1

                # ex edge correction, turn off conductivity in the split field component with a spatial
                # dependence in the y direction

                # don't correct the corners
                # x0 += 1
                # x1 -= 1

                # Ca_0 = (2 * eps - (sig_0 * self.dt)) / (2 * eps + (sig_0 * self.dt))
                # Cb_0 = (2 * self.dt) / ((2 * eps + (sig_0 * self.dt)))

                # get ports that attach to the edges of this PEC face
                x0_ports = []
                x1_ports = []
                y0_ports = []
                y1_ports = []
                for port in self.ports:
                    px, py, pz = port["idx"]
                    # port is normal to the x-axis
                    if port["axis"] == 0:
                        # compare ey indices with the port ez indices, add to overlap list if it overlaps with the
                        # current pec edge. ez and ey have the same x indices
                        if py.start <= y1 and py.stop >= y0:
                            if px.start == x0:
                                x0_ports += [port]
                            elif px.start == x1:
                                x1_ports += [port]

                    # port is normal to the y-axis
                    elif port["axis"] == 1:
                        # compare ex indices with the port ez indices, add to overlap list if it overlaps with the
                        # current pec edge. ez and ex have the same y indices
                        if px.start <= x1 and px.stop >= x0:
                            if py.start == y0:
                                y0_ports += [port]
                            elif py.start == y1:
                                y1_ports += [port]

                # TODO: having an odd number of cells (3) across the PEC trace causes the edge correction to fail,
                # spurious fields appear.
                if edge_correction:
                    # apply correction if no ports attach to the x0 edge
                    if len(x0_ports) == 0:
                        x0, y0, z0 = self.field_pos_to_idx(np.min(pec.points, axis=0), "ey")
                        x1, y1, z1 = self.field_pos_to_idx(np.max(pec.points, axis=0), "ey")
                        # ey edge correction
                        eps = self.eps_ey[x0, y0: y1, z0]
                        self.Ca["ey_x"][x0, y0: y1, z0] = 1
                        self.Cb["ey_x"][x0, y0: y1, z0] = (self.dt / eps)   
                    
                    if len(x1_ports) == 0:
                        x0, y0, z0 = self.field_pos_to_idx(np.min(pec.points, axis=0), "ey")
                        x1, y1, z1 = self.field_pos_to_idx(np.max(pec.points, axis=0), "ey")
                        # ey edge correction
                        eps = self.eps_ey[x0, y0: y1, z0]
                        self.Ca["ey_x"][x1, y0: y1, z0] = 1
                        self.Cb["ey_x"][x1, y0: y1, z0] = (self.dt / eps) 

                    if len(y0_ports) == 0:
                        x0, y0, z0 = self.field_pos_to_idx(np.min(pec.points, axis=0), "ex")
                        x1, y1, z1 = self.field_pos_to_idx(np.max(pec.points, axis=0), "ex")
                        # ex edge correction
                        eps = self.eps_ex[x0: x1, y0, z0]
                        sig = 0
                        self.Ca["ex_y"][x0: x1, y0, z0] = 1
                        self.Cb["ex_y"][x0: x1, y0, z0] = (self.dt / (eps)) 

                        # self.Ca["ex_z"][x0: x1, y0, z0] = 1
                        # self.Cb["ex_z"][x0: x1, y0, z0] = (self.dt / (eps)) 

                        # self.Cb["ey_x"][x0: x1+1, y0-1, z1] *= (dx[x0: x1+1] / (dx[x0: x1+1] * 0.8))

                        # self.Db["hy_x"][x0: x1, y0, z1-1] = self.dt / (2 * u0)
                        # self.Db["hy_x"][x0: x1, y0, z1] = self.dt / (2 * u0)

                        # self.Cb["ez_x"][x0: x1, y0, z1-1] *= (dx[x0: x1] / (dx[x0: x1] * 0.9))
                        # self.Cb["ez_x"][x0: x1, y0, z1] *= (dx[x0: x1] / (dx[x0: x1] * 0.9))

                        # self.Cb["ex_z"][x0: x1, y0, z0] = 2 * (self.dt / eps) 

                        # self.Db["hz_x"][x0: x1, y0-1, z0] = self.dt / (20 * u0)
                        # self.Db["hz_y"][x0: x1, y0-1, z0] = self.dt / (0.05 * u0)

                    if len(y1_ports) == 0:
                        x0, y0, z0 = self.field_pos_to_idx(np.min(pec.points, axis=0), "ex")
                        x1, y1, z1 = self.field_pos_to_idx(np.max(pec.points, axis=0), "ex")
                        # ex edge correction
                        eps = self.eps_ex[x0: x1, y1, z0]
                        sig = 0
                        self.Ca["ex_y"][x0: x1, y1, z0] = 1
                        self.Cb["ex_y"][x0: x1, y1, z0] = (self.dt / eps) 

                        # self.Ca["ex_z"][x0: x1, y1, z0] = 1
                        # self.Cb["ex_z"][x0: x1, y1, z0] = (self.dt / eps) 

                        # self.Cb["ey_x"][x0: x1+1, y1, z1] *= (dx[x0: x1+1] / (dx[x0: x1+1] * 0.8))

                        # self.Db["hy_x"][x0: x1, y1, z1-1] = self.dt / (2 * u0)
                        # self.Db["hy_x"][x0: x1, y1, z1] = self.dt / (2 * u0)

                        # self.Cb["ez_x"][x0: x1, y1, z1-1] *= (dx[x0: x1] / (dx[x0: x1] * 0.9))
                        # self.Cb["ez_x"][x0: x1, y0, z1] *= (dx[x0: x1] / (dx[x0: x1] * 0.9))

                        # self.Cb["ex_z"][x0: x1, y1, z0] = 2 * (self.dt / eps) 

                        # self.Db["hz_x"][x0: x1, y1, z0] = self.dt / (20 * u0)
                        # self.Db["hz_y"][x0: x1, y1, z0] = self.dt / (0.05 * u0)

            else:
                raise NotImplementedError(f"PEC face not supported yet in the given axis.")
            
    def init_ports(self, r0=50):
        """
        
        """
        self.ports = [None] * len(self.port_face)
        
        dx, dy, dz = [conv.m_in(d) for d in self.d_cells]
        dx_h, dy_h, dz_h = [conv.m_in(d) for d in self.dh_cells]
        
        for i, face in enumerate(self.port_face):
    
            if face.n_cells > 1 or face.faces[0] != 4:
                raise ValueError("Only rectangular port faces are supported.")
            
            x0, y0, z0 = self.pos_to_idx(np.min(face.points, axis=0))
            x1, y1, z1 = self.pos_to_idx(np.max(face.points, axis=0))
    
            x0_c, y0_c, z0_c = self.pos_to_idx(np.min(face.points, axis=0), mode="cell")
            x1_c, y1_c, z1_c = self.pos_to_idx(np.max(face.points, axis=0), mode="cell")
    
            # get width of pec in cell units
            idx_d = np.array([x1, y1, z1]) - np.array([x0, y0, z0])
    
            if np.count_nonzero(idx_d) != 2:
                raise ValueError("Port face must be on cartesian grid, and must be 2D.")
            
            # axis that is normal to the port face. Face will have zero width along the normal axis
            axis = np.argmin(idx_d)
                
            # face is normal to the x-axis or y-axis, Resistor is represented by the Ez components
            if (axis == 0) or (axis == 1):
    
                if (axis == 0):
                    # don't include y endpoints in port, the edge of the ms traces use a edge correction method
                    y0 += 1
                    y1 -= 1
                else:
                    # don't include x endpoints in port, the edge of the ms traces use a edge correction method
                    x0 += 1
                    x1 -= 1

                # width of resistor cell centered around the ez component, ez is on the x/y edges
                dx_r = dx_h[x0-1: x1][..., None, None]
                dy_r = dy_h[y0-1: y1][None, :, None]
                # z axis is in center of cell
                dz_r = dz[z0: z1][None, None]
        
                # epsilon of resistor cells
                eps_r = self.eps_ez[x0: x1+1, y0: y1+1, z0: z1]

                # number of components in the xy plane
                n_xy = (eps_r.shape[0] * eps_r.shape[1])
                # resistance of each cell is spilt so the combined resistance of all cells equals r0
                r_cell = r0 * (n_xy) / eps_r.shape[2]
        
                rterm = (r_cell * dx_r * dy_r)
                denom = (eps_r / self.dt) + (dz_r / (2 * rterm))

                ez_idx = tuple([slice(x0, x1+1), slice(y0, y1+1), slice(z0_c, z1_c)])
                
                self.Ca["ez_x"][ez_idx] = ((eps_r / self.dt) - (dz_r / (2 * rterm))) / denom
                self.Cb["ez_x"][ez_idx] = 1 / (denom)
        
                self.Ca["ez_y"][ez_idx] = ((eps_r / self.dt) - (dz_r / (2 * rterm))) / denom
                self.Cb["ez_y"][ez_idx] = 1 / (denom)

                # voltage source coefficient is divided by the number of z components so the total voltage across
                # the port is Vs
                self.ports[i] = dict(idx=ez_idx, Vs_a=-1 / (denom * rterm * eps_r.shape[2]), field="ez", axis=axis)

                # add current probe of current in the termination, normal to the z-axis
                fx0, fy0, fz0 = np.min(face.points, axis=0)
                fx1, fy1, fz1 = np.max(face.points, axis=0)

                iface_z = (fz1 + fz0) / 2
                current_face = pv.Rectangle([
                    [fx0 - 0.001, fy0 - 0.001, iface_z],
                    [fx1 + 0.001, fy0 - 0.001, iface_z],
                    [fx1 + 0.001, fy1 + 0.001, iface_z]
                ])

                self.add_current_probe(f"port{i+1}", current_face)
        
            else:
                raise NotImplementedError(f"Port face normal to the z-axis not implemented yet")

    def add_xPML(self, d_pml=10, side="upper"):
        """
        Add PML layer to the top face of the solution box.
        """
        m_pml = 3 # sigma profile order

        dt = self.dt
        dx = conv.m_in(self.dx[-1]) if side == "upper" else conv.m_in(self.dx[0])
        eta0 = np.sqrt(u0 / e0)
        # now define the values of sigma and sigma_m from the profiles
        sigma_max = 0.8 * (m_pml + 1) / (eta0 * dx)
    
        # define sigma profile in the PML region on the right side of the grid. 
        i_pml = np.arange(0, d_pml)[..., None, None]
    
        # sigma on the cell edges. Components on the edge of the PML have a sigma of 0.
        sigma_e_n = sigma_max * ((i_pml) / (d_pml))**m_pml
        # sigma in the middle of the cells. First Hz component in the PML is 0.5 cells into the PML
        sigma_e_np5 = sigma_max * ((i_pml + 0.5) / (d_pml))**m_pml

        # magnetic conductivity
        # plt.figure()
        # plt.plot(np.arange(0, d_pml, 1), sigma_e_n.squeeze())
        # plt.plot(np.arange(0.5, d_pml + .5, 1), sigma_e_np5.squeeze())

        e_idx = slice(d_pml, 0, -1) if side == "lower" else slice(-d_pml-1, -1)
        h_idx = slice(d_pml-1, None, -1) if side == "lower" else slice(-d_pml, None)

        # ez
        # first ez component is at the edge of the PML where sigma = 0, last component is at the solve boundary 
        # and not updated.
        # sigma / eps must be constant across y and z, page 291 in taflove
        # scale sigma by eps so that sigma / eps is constant
        eps_ez = self.eps_ez[e_idx]
        sigma_ez = np.broadcast_to(sigma_e_n, eps_ez.shape).copy()
        sigma_ez *= (eps_ez / e0)
        
        self.Ca["ez_x"][e_idx] = (2 * eps_ez - (sigma_ez * dt)) / (2 * eps_ez + (sigma_ez * dt))
        self.Cb["ez_x"][e_idx] = (2 * dt) / ((2 * eps_ez + (sigma_ez * dt)))

        # ey
        eps_ey = self.eps_ey[e_idx]
        sigma_ey = np.broadcast_to(sigma_e_n, eps_ey.shape).copy()
        sigma_ey *= (eps_ey / e0)

        self.Ca["ey_x"][e_idx] = (2 * eps_ey - (sigma_ey * dt)) / (2 * eps_ey + (sigma_ey * dt))
        self.Cb["ey_x"][e_idx] = (2 * dt) / ((2 * eps_ey + (sigma_ey * dt)))

        # hx/hy components are in the middle of the PML cells, use half cell indices
        eps_hy = self.eps_hy[h_idx]
        simga_e_hy = np.broadcast_to(sigma_e_np5, (d_pml,) + self.Da["hy_x"].shape[1:]).copy()
        simga_e_hy *= (eps_hy / e0)
        sigma_m_hy = simga_e_hy * u0 / eps_hy

        self.Da["hy_x"][h_idx] = (2 * u0 - (sigma_m_hy * dt)) / (2 * u0 + (sigma_m_hy * dt))
        self.Db["hy_x1"][h_idx] = (2 * dt) / ((2 * u0 + (sigma_m_hy * dt))) 
        self.Db["hy_x2"][h_idx] = (2 * dt) / ((2 * u0 + (sigma_m_hy * dt))) 

        eps_hz = self.eps_hz[h_idx]
        sigma_e_hz = np.broadcast_to(sigma_e_np5, (d_pml,) + self.Da["hz_x"].shape[1:]).copy()
        sigma_e_hz *= (eps_hz / e0)
        sigma_m_hz = sigma_e_hz * u0 / eps_hz

        self.Da["hz_x"][h_idx] = (2 * u0 - (sigma_m_hz * dt)) / (2 * u0 + (sigma_m_hz * dt))
        self.Db["hz_x1"][h_idx] = (2 * dt) / ((2 * u0 + (sigma_m_hz * dt)))
        self.Db["hz_x2"][h_idx] = (2 * dt) / ((2 * u0 + (sigma_m_hz * dt)))


    def run(self, ports, v_waveforms, n_threads=4):

        if isinstance(ports, int):
            ports = [ports]

        v_waveforms = np.atleast_2d(v_waveforms)
            
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

        # dx with an extra component for the dummy end cell
        dx1_h_inv = np.vstack([dx_h_inv, [[[0]]]])

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

        print(f"Running solver with {self.Nx * self.Ny * self.Nz / 1e3:.1f}k cells")
        stime = time.time()
        core.core_func.solver_run(coefficients, probes, monitors, mem, Nx, Ny, Nz, Nt, n_threads)
        print(f"Done in {time.time() - stime:.3f}s")

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


    def add_field_monitor(self, name: str, field: str, axis: str, position: float, n_step: int):
        """
        Add field monitor along a slice through the 3D volume

        Parameters
        ----------
        name : str
        field : {'ex', 'ey', 'ez', 'hx', 'hy', 'hz'}
        axis : {'x', 'y', 'z'}
        position : int
            index on axis of the slice
        t_step : int, default: 1
            number of time steps between each capture.
        """

        if field not in self.fshape.keys():
            raise ValueError(f"Unsupported field: {field}. Expecting one of: {tuple(self.fshape.keys())}")
        if axis not in ('x', 'y', 'z'):
            raise ValueError(f"Unsupported axis: {axis}")

        # get the spatial shape of the field slice
        axis_i = dict(x=0, y=1, z=2)[axis]
        shape = list(self.fshape[field])
        axis_len = shape.pop(axis_i)

        # convert position to field index
        full_pos = [0] * 3
        full_pos[axis_i] = position
        idx = self.field_pos_to_idx(full_pos, field)

        if idx[axis_i] >= (axis_len - 1):
            raise ValueError("Field position out of bounds")

        self.monitors[name] = dict(
            field=field, axis=axis_i, position=position, index=int(idx[axis_i]), n_step=n_step, shape=tuple(shape)
        )

    def add_probe(self, name: str, field: str, position: tuple):
        """
        Add field monitor along a slice through the 3D volume

        Parameters
        ----------
        name : str
        field : {'ex', 'ey', 'ez', 'hx', 'hy', 'hz'}
        position : tuple(int, int, int)
            index of field component in the grid
        """

        if field not in self.fshape.keys():
            raise ValueError(f"Unsupported field: {field}. Expecting one of: {tuple(self.fshape.keys())}")

        idx = self.field_pos_to_idx(position, field)

        if any([p >= (fs - 1) for p, fs in zip(idx, self.fshape[field])]):
            raise ValueError("Field position out of bounds")
        
        self.probes[name] = dict(
            field=field, position=position, index=idx
        )

    def add_current_probe(self, name: str, face: pv.PolyData):
        """
        
        """
        dx_h, dy_h, dz_h = [conv.m_in(d) for d in self.dh_cells]

        if face.n_cells > 1 or face.faces[0] != 4:
            raise ValueError("Only rectangular current faces are supported.")
                
        # get axis that face is constant over (the normal axis)
        axis = np.argmin(np.any(np.diff(face.points, axis=0), axis=0))

        # minimum and maximum extents of current face
        pmin, pmax = np.min(face.points, axis=0), np.max(face.points, axis=0)

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
        
    def add_voltage_probe(self, name: str, line: pv.PolyData):
        """
        
        """
        dx, dy, dz = [conv.m_in(d) for d in self.d_cells]

        if len(line.faces) > 1:
            raise ValueError("Only line voltage probes are supported.")
                
        # get axis that face is constant over (the normal axis)
        axis = np.argmax(np.any(np.diff(line.points, axis=0), axis=0))

        # minimum and maximum extents of current face
        pmin, pmax = np.min(line.points, axis=0), np.max(line.points, axis=0)

        if axis == 2: # voltage is along z-axis
            # all components have the same x-index and y-index
            xyz_min = self.field_pos_to_idx(pmin, "ez")
            xyz_max = self.field_pos_to_idx(pmax, "ez")

            # add probes along the axis between both endpoints
            i = 0
            for z in np.arange(xyz_min[2], xyz_max[2]):
                self.probes[f"{name}_{i}"] = dict(field="ez", index=(xyz_min[0], xyz_min[1], z), d=-dz[z])
                i += 1

        else:
            raise NotImplementedError("Voltage probe not supported in the given direction yet")
      
    def render(self, show_probes=False, point_size=15) -> pv.Plotter:
        """
        Plot the model geometry
        """

        gx, gy, gz = self.g_edges
        gx_h, gy_h, gz_h = self.g_cells
        
        grid = pv.RectilinearGrid(gx, gy, gz)
        
        plotter = pv.Plotter()
        plotter.enable_parallel_projection()
        # add grid
        plotter.add_mesh(grid, style="wireframe", line_width=0.05, color="k", opacity=0.05)

        # add substrates
        for name, (sub, er) in self.substrate.items():
            plotter.add_mesh(sub, **self.styles[name])
            
        # add pec
        for name, pec in self.pec_face.items():
            plotter.add_mesh(pec, **self.styles[name])

        # add ports
        for port_face in self.port_face:
            plotter.add_mesh(port_face, color="pink", opacity=0.5)

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

    def plot_cooeficients(
        self, field, value, axis, position, vmin=None, vmax=None, opacity=1, cmap="jet", point_size=10, normalization=None
    ):
            
        plotter = self.render()

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

    def vi_probe_values(self, name: str):
        """
        Returns time domain current or voltage for the given probe.
        """
        return np.sum([p["values"] * p["d"] for k, p in self.probes.items() if k[:len(name)] == name], axis=0)

    def get_sparameters(self, frequency, source_port=1, z0=50):
        """
        Returns a column of the sparameter matrix excited by a single port.
        """
        nports = len(self.ports)
        nfrequency = len(frequency)

        # exiting waves (B) from each port
        B = np.zeros((nfrequency, nports), dtype=np.complex128)

        ports = np.arange(1, nports+1)
        z_idx = self.ports[source_port-1]["idx"][2]

        # source port voltage at each component, shape is x, y, z, time
        src_component_v = self.ports[source_port-1]["values"] * (conv.m_in(self.dz[z_idx])[None, None, :, None])

        # add voltage along z
        src_voltage = -np.sum(src_component_v, axis=2)
        # average voltage across edge of port, if port is normal to y this is the x axis, if normal to x, average
        # along y
        src_axis = self.ports[source_port-1]["axis"]
        src_vp = np.mean(src_voltage, axis=1 if src_axis == 0 else 0).squeeze()
        # source port applied voltage
        src_applied = self.ports[source_port-1]["src"]

        i_s = self.vi_probe_values(f"port{source_port}")
        Is = utils.dtft(i_s, frequency, 1 / self.dt)

        Vs = utils.dtft(src_vp, frequency, 1 / self.dt)

        # delay current by half a time-step to be at the same time sample as the voltage
        # h components are ahead of the e components by half a time step
        # Is = Is * np.exp(1j * 2 * np.pi * frequency * self.dt / 2)

        # As = (Vs + z0 * Is) / (2 * np.sqrt(z0.real))
        # Bs = (Vs - np.conj(z0) * Is) / (2 * np.sqrt(z0.real))

        # B[:, source_port-1] = Bs


        # source port S11, reflected wave (b) is the difference of the total voltage across the port,
        # and the incident wave V = a + b
        As = utils.dtft(src_applied, frequency, 1 / self.dt)
        V = utils.dtft(src_vp, frequency, 1 / self.dt)
        B[:, source_port-1] = V - (As)


        # the exiting waves on other ports is the voltage that appears across the terminations. Note this is
        # different than the total voltage across the port because that is the sum of the reflected wave that
        # doesn't make it into the load, and the wave transmitted through the load.
        exit_ports = np.array([p for p in ports if p != source_port])
        for p in exit_ports:

            # voltage generated by termination due the current through it. 
            vp = -self.vi_probe_values(f"port{p}") * z0
            # h-fields are 1/2 time step ahead of the e-fields. Dely current so they are at the same time step
            Vp = utils.dtft(vp, frequency, 1 / self.dt) * np.exp(-1j * frequency * 2 * np.pi * (self.dt / 2))
            # populate row of the exiting wave matrix
            B[:, p-1] = Vp
        
        # return a single column of the full s-matrix
        return B / As[..., None]
    
    def plot_monitor(
        self,
        monitor_name: list,
        view="xz",
        vmax=None, vmin=None,
        zoom=1.3, 
        el=10, az=0,
        opacity="linear",
        gif_file=None,
        linear=False, 
        cmap="jet",
        style="points"
    ):

        monitor_name = np.atleast_1d(monitor_name)
        plotter = self.render()
        field_meshes = []
        field_actors = []

        for i, m_name in enumerate(monitor_name):

            monitor = self.monitors[m_name]
            
            field = monitor["values"]
            axis = monitor["axis"]
            
            n_step = monitor["n_step"]

            slc = [slice(None)] * 3
            slc[axis] = monitor["index"]

            field = np.where(np.abs(field) < 1e-12, 1e-12, field)
            if linear:
                field_v = field
            else:
                field_v = 20 * np.log10(np.abs(field))

            if vmax is None:
                vmax = np.nanmax(field_v)
            if vmin is None:
                vmin = -vmax if linear else vmax - 50

            field_v = np.clip(field_v, vmin, vmax).reshape(len(field), -1, order="F")
        
            nframe = len(field)

            floc_in = self.floc[monitor["field"]]

            # slice each axis of grid locations, forcing integer indices into slices
            g = [floc_in[i][s] if isinstance(s, slice) else floc_in[i][s: s+1] for i, s in enumerate(slc)]
            
            fmesh = pv.RectilinearGrid(*g)
            fmesh.point_data['values'] = field_v[0]
            
            actor = plotter.add_mesh(
                fmesh, 
                cmap=cmap, 
                scalars="values", 
                clim=[vmin, vmax], 
                show_scalar_bar=False, 
                opacity=opacity[i],
                interpolate_before_map=style == "surface",
                render_points_as_spheres=True,
                style=style,
                lighting=False,
                point_size=10
            )

            field_meshes += [(fmesh, field_v)]
            field_actors += [actor]

        plotter.add_axes()
        plotter.camera_position = view
        plotter.camera.elevation += el
        plotter.camera.azimuth += az
        plotter.camera.zoom(zoom)
        
        plotter.add_scalar_bar(
            title="E [dB]\n" if linear else "E [V/m]\n", vertical=False, label_font_size=11, title_font_size=14
        )
        
        Nt = nframe * n_step

        def callback(nt):
            
            # downsampled index into the field values
            n = int(nt // n_step)

            self.slider_value = n

            for m, f in field_meshes:
                m.point_data["values"][:] = f[n]
          
        self.slider_value = Nt // 2
        self.slider = plotter.add_slider_widget(
            callback,
            [0, Nt-1],
            value=Nt // 2,
            title="Time Step",
            interaction_event="always",
            style="modern",
            fmt="%0.0f"
        )

        def set_field_visible(value):
            for m in field_actors:
                m.SetVisibility(value)


        def increment_left():
            self.slider.GetRepresentation().SetValue(self.slider_value - 1)
            callback(self.slider_value - 1)
            plotter.render()

        def increment_right():
            self.slider.GetRepresentation().SetValue(self.slider_value + 1)
            callback(self.slider_value + 1)
            plotter.render()

        # add checkbox to turn off field visibility
        plotter.add_checkbox_button_widget(set_field_visible, value=True, position=(10, 10), size=30, border_size=0)
        plotter.add_text("Field Visibility", position=(45, 15), font_size=9)

        plotter.add_key_event("Left", increment_left)
        plotter.add_key_event("Right", increment_right)


        if gif_file:
            plotter.open_gif(gif_file)
            for n in range(nframe):
                for m, f in field_meshes:
                    m.point_data["values"][:] = f[n].flatten(order="F")
                    
                plotter.add_title(f"n={n * n_step}")
                # plotter.render()
                plotter.write_frame()

            # Closes and finalizes movie
            plotter.close()
        
        return plotter
    


