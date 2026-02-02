import itertools
import time
import numpy as np 
from scipy import signal
import pyvista as pv

from rfnetwork import const, conv, utils, core
from . solver import Solver_3D

u0 = const.u0
e0 = const.e0
c0 = const.c0

class Solver_PCB(Solver_3D):

    def init_mesh(self, d0, n0, d_pec, n_min_pec, d_sub, n_min_sub, blend_pec = False):
        """
        Initialize the spatial grid.
        """
        edges = [np.array([], dtype=np.float32) for i in range(3)]

        objects = [self.bounding_box] + list(self.conductor.values()) + [sub["obj"] for sub in self.dielectric.values()]
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
        for sub in self.substrate.values():
            sub_z_max = np.max([sub_z_max, np.max(sub["obj"].points[:, 2])])
            sub_er_max = np.max([sub_er_max, sub["er"]])

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
        self.sigma = np.zeros(self.n_cells)

        for sub in self.dielectric.values():
            x0, y0, z0 = self.pos_to_idx(np.min(sub["obj"].points, axis=0), mode="cell")
            x1, y1, z1 = self.pos_to_idx(np.max(sub["obj"].points, axis=0), mode="cell")
        
            self.eps[x0: x1, y0: y1, z0: z1] = e0 * sub["er"]
            # non-dispersive conductivity at a single frequency
            self.sigma[x0: x1, y0: y1, z0: z1] = sub["loss_tan"] * e0 * sub["er"] * 2 * np.pi * sub["f0"]

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

    def init_pec(self, edge_correction=False):
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
                    # if len(x0_ports) == 0:
                    #     x0, y0, z0 = self.field_pos_to_idx(np.min(pec.points, axis=0), "ey")
                    #     x1, y1, z1 = self.field_pos_to_idx(np.max(pec.points, axis=0), "ey")
                    #     # ey edge correction
                    #     eps = self.eps_ey[x0, y0: y1, z0]
                    #     self.Ca["ey_x"][x0, y0: y1, z0] = 1
                    #     self.Cb["ey_x"][x0, y0: y1, z0] = (self.dt / eps)   
                    
                    # if len(x1_ports) == 0:
                    #     x0, y0, z0 = self.field_pos_to_idx(np.min(pec.points, axis=0), "ey")
                    #     x1, y1, z1 = self.field_pos_to_idx(np.max(pec.points, axis=0), "ey")
                    #     # ey edge correction
                    #     eps = self.eps_ey[x0, y0: y1, z0]
                    #     self.Ca["ey_x"][x1, y0: y1, z0] = 1
                    #     self.Cb["ey_x"][x1, y0: y1, z0] = (self.dt / eps) 


                    if len(y0_ports) == 0:
                        x0, y0, z0 = self.field_pos_to_idx(np.min(pec.points, axis=0), "ex")
                        x1, y1, z1 = self.field_pos_to_idx(np.max(pec.points, axis=0), "ex")

                        # self.Ca["ex_y"][x0: x1, y0, z0] = 1
                        # self.Cb["ex_y"][x0: x1, y0, z0] = (self.dt / (eps)) 
                        # self.Ca["ex_z"][x0: x1, y0, z0] = 1
                        # self.Cb["ex_z"][x0: x1, y0, z0] = (self.dt / (eps)) 

                        # self.Db["hy_z2"][x0: x1, y0, z0-1] = 0
                        # self.Db["hy_z1"][x0: x1, y0, z0] = 0

                        # self.Db["hz_y1"][x0: x1, y0-1, z0] *= hy_CF
                        # self.Db["hz_y2"][x0: x1, y0-1, z0] *= hy_CF
                        CF = 2 * np.sqrt(1/2)
                        # hz in the same plane as the trace
                        self.Db["hz_y2"][x0: x1, y0-1, z0] *= 1 / CF
                        self.Db["hz_y1"][x0: x1, y0-1, z0] *= 1 / CF

                        # hy directly above and below trace edge
                        self.Db["hy_z1"][x0: x1, y0, z0-1] *= 1 / CF
                        self.Db["hy_z2"][x0: x1, y0, z0-1] *= 1 / CF
                        self.Db["hy_z1"][x0: x1, y0, z0] *= 1 / CF
                        self.Db["hy_z2"][x0: x1, y0, z0] *= 1 / CF

                        # hx below the trace, correct ez component on the edge
                        self.Db["hx_y2"][x0+1: x1-1, y0-1, z0-1] *= CF
                        self.Db["hx_y1"][x0+1: x1-1, y0, z0-1] *= CF
                        # hy above the trace, correct the ez component on the edge
                        self.Db["hx_y2"][x0+1: x1-1, y0-1, z0] *= CF
                        self.Db["hx_y1"][x0+1: x1-1, y0, z0] *= CF

                        # ez below and above the trace
                        self.Cb["ez_y"][x0+1: x1-1, y0, z0] *= 1/0.785
                        self.Cb["ez_y"][x0+1: x1-1, y0, z0-1] *= 1/0.785

                        # # # ey in the plane of the trace
                        # self.Cb["ey_z"][x0: x1, y0-1, z0] *= 1/0.785

                    

                    if len(y1_ports) == 0:
                        x0, y0, z0 = self.field_pos_to_idx(np.min(pec.points, axis=0), "ex")
                        x1, y1, z1 = self.field_pos_to_idx(np.max(pec.points, axis=0), "ex")

                        CF = 2 * np.sqrt(1/2)

                        # hz in the same plane as the trace
                        self.Db["hz_y1"][x0: x1, y1, z0] *= 1 / CF
                        self.Db["hz_y2"][x0: x1, y1, z0] *= 1 / CF

                        # hy directly above and below trace edge
                        self.Db["hy_z1"][x0: x1, y1, z0-1] *= 1 / CF
                        self.Db["hy_z2"][x0: x1, y1, z0-1] *= 1 / CF
                        self.Db["hy_z1"][x0: x1, y1, z0] *= 1 / CF
                        self.Db["hy_z2"][x0: x1, y1, z0] *= 1 / CF

                        # hy below the trace, correct ez component on the edge
                        self.Db["hx_y2"][x0+1: x1-1, y1-1, z0-1] *= CF
                        self.Db["hx_y1"][x0+1: x1-1, y1, z0-1] *= CF
                        # hy above the trace, correct the ez component on the edge
                        self.Db["hx_y2"][x0+1: x1-1, y1-1, z0] *= CF
                        self.Db["hx_y1"][x0+1: x1-1, y1, z0] *= CF

                        # # ez below and above the trace
                        self.Cb["ez_y"][x0+1: x1-1, y1, z0-1] *= 1/0.785
                        self.Cb["ez_y"][x0+1: x1-1, y1, z0] *= 1/0.785

                        # # # ey in the plane of the trace
                        # self.Cb["ey_z"][x0: x1, y1, z0] *= 1/0.785

            else:
                raise NotImplementedError(f"PEC face not supported yet in the given axis.")