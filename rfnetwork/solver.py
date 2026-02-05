import itertools
import time
import numpy as np 
from scipy import signal
import pyvista as pv
from copy import copy
from np_struct import ldarray


from rfnetwork import const, conv, utils, core

u0 = const.u0
e0 = const.e0
c0 = const.c0


class Solver_3D():
    """
    3D EM Solver for PCB geometries, substrates are normal to the z-axis.
    """

    def __init__(self, bounding_box):
        self.dielectric = dict()
        self.conductor = dict()
        self.pml_boundaries = []

        self.port_face = []
        self.bounding_box = bounding_box
        self.monitors=dict()
        self.probes=dict()
        self.slider_value = 0
        self.n_pml = None

        self.sbox_max = np.max(bounding_box.points, axis=0)
        self.sbox_min = np.min(bounding_box.points, axis=0)

        self._places = 5 # hundredth of a mil
        self._tol = 1 / (10 ** self._places)

    def add_dielectric(self, name, obj, er: float, loss_tan=0, f0=0, style: dict = dict()):
        """
        Add rectangular dielectric
        """
        if obj.n_cells > 6:
            raise NotImplementedError("Only rectangular dielectrics are supported.")
        
        self.dielectric[name] = dict(obj=obj, er=er, loss_tan=loss_tan, f0=f0, style=style)

    def add_conductor(self, name, obj, sigma=1e16, style: dict = dict()):
        """
        Add rectangular conductor
        """
        if obj.n_cells > 6:
            raise NotImplementedError("Only rectangular conductors are supported.")
        
        self.conductor[name] = dict(obj=obj, sigma=sigma, style=style)

    def add_lumped_port(self, number, face):
        """
        Attach a lumped port to a face
        """
        # add new ports to the faces list
        if len(self.port_face) < number:
            self.port_face = self.port_face + [None] * (number - len(self.port_face))

        self.port_face[number - 1] = face

    def assign_PML_boundaries(self, *sides: str, n_pml: float = 10):
        """
        Assign a PML boundary to sides of the solve box.

        Parameters
        ----------
        sides : list, str
            Valid values are ("x+", "x-", "y+", "y-", "z+", "z-",)
        """

        valid_sides = ("x+", "x-", "y+", "y-", "z+", "z-")
        if any([s not in valid_sides for s in sides]):
            raise ValueError(f"PML side not recognized. Expecting one of {valid_sides}")

        self.pml_boundaries = copy(list(sides))
        self.n_pml = n_pml

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
            # equal to or larger than the point.
            idx += [np.argmax(diff >= -self._tol) if diff[-1] > -self._tol else len(g)]
        
        return tuple(idx)
    
    def generate_mesh(self, d0: float, d_edge: float = None, z_bounds: float = None):
        """
        
        """
        self._init_grid(d0=d0, d_edge=d_edge, z_bounds=z_bounds)
        # init dielectrics sets the cell sigma and er but does not set the coefficients
        self._init_dielectrics()
        # initialize coefficients from the cell sigma and er set by the dielectrics
        self._init_coefficients()

        # add PML layers before conductors and ports, this gives priority to any conductors that may extend into the 
        # PML
        for pml_side in self.pml_boundaries:
            self._init_PML(pml_side)

        self._init_conductors()
        self.init_ports()

                
    def _init_grid(self, d0: float, d_edge: float = None, z_bounds: float = None):
        """
        Initialize the spatial grid.
        """

        if not isinstance(d0, list):
            d0 = [d0] * 3

        d_bounds = [[d0[i], d0[i]] for i in range(3)]

        if z_bounds is not None:
            d_bounds[2] = z_bounds

        edges = [np.array([], dtype=np.float32) for i in range(3)]

        objects = [self.bounding_box] + [cond["obj"] for cond in self.conductor.values()] + [sub["obj"] for sub in self.dielectric.values()]
        dtype_ = np.float32

        # build list of edge coordinates along each axis
        for obj in objects:
            # round points to minimum precision supported by the mesh
            p_edges = np.around(obj.points.T, decimals=self._places).astype(np.float32)
        
            for i in range(3):
                edges_i = np.unique(np.concatenate([edges[i], p_edges[i]]))
                edges[i] = edges_i

        # build a list of cell widths along each axis
        mesh_cells_d = [[], [], []]
        for axis in range(3):

            # list of distances between each edge
            cells_d = np.diff(edges[axis])

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

                # add an edge cell at the end of the the last cell
                if d_edge is not None and (i > 0) and (subcells_d[-1] > (d_edge * 1.5)): 
                    # make the last cell shorter to compensate for adding a new d_edge cell
                    subcells_d[-1] -= d_edge
                    # add new sub-cell at the edge
                    subcells_d += [d_edge]

                # add an edge cell at the beginning of the current cell
                if d_edge is not None and (i > 0) and (subcells_d_i[0] > (d_edge * 1.5)):
                    # make the subcell next to the edge shorter 
                    subcells_d_i[0] -= d_edge
                    # add new sub-cell at the edge
                    subcells_d_i = ([d_edge] + subcells_d_i)

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
                # If on the edge of the grid, match to the default d0
                else:
                    dprev = d_bounds[axis][0]

                # next cell width
                if (i < len(subcells_d) - 1) and (graded_subcells_d[i + 1] is not None):
                    dnext = graded_subcells_d[i + 1][0]
                elif (i < len(subcells_d) - 1) :
                    dnext = np.clip(subcells_d[i + 1], 0, d0[axis] )
                # If on the edge of the grid, match to the boundary width
                else:
                    dnext = d_bounds[axis][1]

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

        for cond in self.conductor.values():

            obj = cond["obj"]
            sig = cond["sigma"]
            
            # get indices of the grid edges that bound the conductor
            x0, y0, z0 = self.pos_to_idx(np.min(obj.points, axis=0), mode="edge")
            x1, y1, z1 = self.pos_to_idx(np.max(obj.points, axis=0), mode="edge")

            # ex components in the conductor. The y component is the middle of the cell and is only included if the
            # conductor extends to both edges on either side of it.
            ex_idx = (slice(x0, x1), slice(y0, y1+1), slice(z0, z1+1))

            Ca_ex = (2 * self.eps_ex[ex_idx] - (sig * self.dt)) / (2 * self.eps_ex[ex_idx] + (sig * self.dt))
            Cb_ex = (2 * self.dt) / ((2 * self.eps_ex[ex_idx] + (sig * self.dt)))
            self.Ca["ex_y"][ex_idx] = Ca_ex
            self.Cb["ex_y"][ex_idx] = Cb_ex
            self.Ca["ex_z"][ex_idx] = Ca_ex
            self.Cb["ex_z"][ex_idx] = Cb_ex
            
            # ey components in the conductor. The y component is the middle of the cell and is only included if the
            # conductor extends to both edges on either side of it.
            ey_idx = (slice(x0, x1+1), slice(y0, y1), slice(z0, z1+1))

            Ca_ey = (2 * self.eps_ey[ey_idx] - (sig * self.dt)) / (2 * self.eps_ey[ey_idx] + (sig * self.dt))
            Cb_ey = (2 * self.dt) / ((2 * self.eps_ey[ey_idx] + (sig * self.dt)))
            self.Ca["ey_z"][ey_idx] = Ca_ey
            self.Cb["ey_z"][ey_idx] = Cb_ey
            self.Ca["ey_x"][ey_idx] = Ca_ey
            self.Cb["ey_x"][ey_idx] = Cb_ey

            # ez components in the conductor. The z component is the middle of the cell and is only included if the
            # conductor extends to both edges on either side of it.
            ez_idx = (slice(x0, x1+1), slice(y0, y1+1), slice(z0, z1))

            Ca_ez = (2 * self.eps_ez[ez_idx] - (sig * self.dt)) / (2 * self.eps_ez[ez_idx] + (sig * self.dt))
            Cb_ez = (2 * self.dt) / ((2 * self.eps_ez[ez_idx] + (sig * self.dt)))
            self.Ca["ez_x"][ez_idx] = Ca_ez
            self.Cb["ez_x"][ez_idx] = Cb_ez
            self.Ca["ez_y"][ez_idx] = Ca_ez
            self.Cb["ez_y"][ez_idx] = Cb_ez


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



    def edge_correction(self, edge):
        # TODO: having an odd number of cells (3) across the PEC trace causes the edge correction to fail,
        # spurious fields appear.
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
    
                # if (axis == 0):
                #     # don't include y endpoints in port, the edge of the ms traces use a edge correction method
                #     y0 += 1
                #     y1 -= 1
                # else:
                #     # don't include x endpoints in port, the edge of the ms traces use a edge correction method
                #     x0 += 1
                #     x1 -= 1

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

    def run(self, ports, v_waveforms, n_threads=4):

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

        print(f"Running solver with {self.Nx * self.Ny * self.Nz / 1e3:.1f}k cells, and {Nt} time steps.")
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
        position : float
            position on axis of the slice
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
        idx = int(self.field_pos_to_idx(full_pos, field)[axis_i])

        if idx >= (axis_len - 1):
            raise ValueError("Field position out of bounds")

        self.monitors[name] = dict(
            field=field, axis=axis_i, position=position, index=idx, n_step=n_step, shape=tuple(shape)
        )

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
        for name, (sub) in self.dielectric.items():
            plotter.add_mesh(sub["obj"], **sub["style"])
            
        # add pec
        for name, cond in self.conductor.items():
            plotter.add_mesh(cond["obj"], **cond["style"])

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

    def plot_coefficients(
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
    
    def line_probe_values(self, name: str):

        return np.array([p["values"] for k, p in self.probes.items() if k[:len(name)] == name])

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

        # i_s = self.vi_probe_values(f"port{source_port}")
        # Is = utils.dtft(i_s, frequency, 1 / self.dt)

        # Vs = utils.dtft(src_vp, frequency, 1 / self.dt)

        # delay current by half a time-step to be at the same time sample as the voltage
        # h components are ahead of the e components by half a time step
        # Is = Is * np.exp(1j * 2 * np.pi * frequency * self.dt / 2)

        # As = (Vs + z0 * Is) / (2 * np.sqrt(z0.real))
        # Bs = (Vs - np.conj(z0) * Is) / (2 * np.sqrt(z0.real))

        # B[:, source_port-1] = Bs

        # freq = np.fft.fftfreq(len(src_vp), self.dt)
        # V = np.fft.fft(src_vp)
        frs = np.max(frequency) * 4
        fs = 1 / self.dt

        # downsample the time-domain waveforms, in most cases 1 / self.dt is much greater than the highest
        # frequency of interest. Down-sampling makes the FFT or DTFT much faster
        if fs > (frs * 2):

            sos1 = signal.butter(20, frs/2, btype="lowpass", output="sos", fs = fs)
            src_vp_f = signal.sosfilt(sos1, src_vp)
            src_applied_f = signal.sosfilt(sos1, src_applied)

            # downsample
            downsample_factor = int(fs / frs)
            src_vp_rs = src_vp_f[::downsample_factor]
            src_applied_rs = src_applied_f[::downsample_factor]

            # filter again above fmax
            sos2 = signal.butter(20, np.max(frequency) * 1.5, btype="lowpass", output="sos", fs = frs)
            src_vp = signal.sosfilt(sos2, src_vp_rs)
            src_applied = signal.sosfilt(sos2, src_applied_rs)

        # source port S11, reflected wave (b) is the difference of the total voltage across the port,
        # and the incident wave V = a + b
        As = utils.dtft(src_applied, frequency, frs)
        V = utils.dtft(src_vp, frequency, frs)
        B[:, source_port-1] = V - (As)

        # the exiting waves on other ports is the voltage that appears across the terminations. Note this is
        # different than the total voltage across the port because that is the sum of the reflected wave that
        # doesn't make it into the load, and the wave transmitted through the load.
        exit_ports = np.array([p for p in ports if p != source_port])
        for p in exit_ports:

            # voltage generated by termination due the current through it. 
            vp = -self.vi_probe_values(f"port{p}") * z0

            # downsample the time-domain data 
            if fs > (frs * 2):
                vp_f = signal.sosfilt(sos1, vp)
                vp_rs = vp_f[::downsample_factor]
                vp = signal.sosfilt(sos2, vp_rs)

            # h-fields are 1/2 time step ahead of the e-fields. Dely current so they are at the same time step
            Vp = utils.dtft(vp, frequency, frs) #* np.exp(-1j * frequency * 2 * np.pi * (self.dt / 2))
            # populate row of the exiting wave matrix
            B[:, p-1] = Vp
        
        # return a single column of the full s-matrix
        return B / As[..., None]
    
    def get_monitor_data(self, name):
        """
        
        """
        monitor = self.monitors[name]
        values = monitor["values"]
        t_len = len(values) * self.dt * monitor["n_step"]

        time_values = np.arange(0, t_len, self.dt * monitor["n_step"], dtype=np.float64)

        # build coordinates in inches for the two spatial dimensions of the slice
        spatial_axis = [0, 1, 2]
        spatial_dims = ["x", "y", "z"]
        spatial_axis.pop(monitor["axis"])
        spatial_coords = {spatial_dims[i]: self.floc[name][i] for i in spatial_axis}

        return ldarray(
            monitor["values"], 
            coords=dict(time=time_values, **spatial_coords)
        )

 
    
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
            [0, Nt-2],
            value=Nt // 2,
            title="Time [ps]",
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
    


