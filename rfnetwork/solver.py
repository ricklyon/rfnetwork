import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time

from IPython.display import Image as ipyimage
import rfnetwork as rfn
import mpl_markers as mplm
import matplotlib.colors as mcolors
from core import core_func

import sys
import matplotlib
from itertools import product
# matplotlib.use("qt5agg")

pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)

sys.argv = sys.argv[0:1]

u0 = const.u0
e0 = const.e0
c0 = const.c0



class SolverMesh():

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
            idx += [np.argmax(diff >= -1e-3) if diff[-1] > 0 else len(g)]

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
            idx += [np.argmax(diff >= -1e-3) if diff[-1] > 0 else len(g)]
        
        return tuple(idx)
    
    def init_grid(self, d0, n_feature_min=2):
        """
        Initialize the spatial grid.
        """
        edges = [np.array([], dtype=np.float32) for i in range(3)]
        objects = [self.bounding_box] + list(self.pec_face.values()) + [sub[0] for sub in self.substrate.values()]
        dtype_ = np.float32

        for obj in objects:
            # round points to minimum precision supported by the mesh
            p_edges = np.around(obj.points.T, decimals=3).astype(np.float32)
        
            for i in range(3):
                edges[i] = np.unique(np.concatenate([edges[i], p_edges[i]]))

        # list of cell widths for each axis
        cell_d = [[], [], []]
        # iterate over x, y, z axis
        for i, d0_i in enumerate(d0):
            edges_i = edges[i]
        
            # list of distances between each edge
            d_e = np.diff(edges_i)
            
            for d in d_e:
                # how many cells of the default size will fit in this interval
                nx = d / d0_i
            
                # use a minimum of two cells between adjacent features
                if nx < n_feature_min:
                    cell_d[i] += [d / n_feature_min] * n_feature_min
            
                else:
                    # make the cell a bit smaller than the default to account for the remainder of nx
                    nx = int(nx + 1) if nx % 1 > 1e-3 else int(nx)
                    # d0 * nx will be larger than d if there was a remainder on nx, divide up this difference amoung each cell
                    d0_e = d0_i - (((d0_i * nx) - d) / nx)
                    cell_d[i] += [d0_e] * nx
            
        # TODO: blend large differences in adjacent cell widths
        
        gx, gy, gz = [np.around(np.concatenate([[self.sbox_min[i]], self.sbox_min[i] + np.cumsum(cell_d[i])]), decimals=6) for i in range(3)]
        
        self.grid_mesh = pv.RectilinearGrid(gx, gy, gz)
        # cell widths
        dx, dy, dz = np.diff(gx).astype(dtype_), np.diff(gy), np.diff(gz)
        # number of cells along each axis
        self.n_cells = len(dx), len(dy), len(dz)

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
        


    def mesh(self, d0, n_feature_min=2):
        """
        Build FDTD coefficients for all grid cells.
        """
        s.init_grid(d0, n_feature_min)
        dtype_ = np.float32
        
        dx, dy, dz = [conv.m_in(d) for d in self.d_cells]
        Nx, Ny, Nz = self.n_cells
        
        # compute maximum time step that ensures convergence, use freespace propagation speed as worst case
        length_min = np.array([np.min(dx), np.min(dy), np.min(dz)])
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

        self.eps_ex = np.ones(self.fshape["ex"], dtype=dtype_) * e0
        self.eps_ey = np.ones(self.fshape["ey"], dtype=dtype_) * e0
        self.eps_ez = np.ones(self.fshape["ez"], dtype=dtype_) * e0

        # ex component is on the y and z edge of the cell
        self.eps_ex[:, 1:-1, 1:-1] = ((eps_y[..., :-1] * (dz0/2) + eps_y[..., 1:] * (dz1/2)) / (dz0/2 + dz1/2))
        # ey component is on the x and z edge of the cell
        self.eps_ey[1:-1, :, 1:-1] = ((eps_x[..., :-1] * (dz0/2) + eps_x[..., 1:] * (dz1/2)) / (dz0/2 + dz1/2))
        # ez component is on the x and y edge of the cell, average eps from adjacent cells on both axis
        self.eps_ez[1:-1, 1:-1] = ((eps_x[:, :-1] * (dy0/2) + eps_x[:, 1:] * (dy1/2)) / (dy0/2 + dy1/2))

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

        self.Da = dict(
            hx_y = np.ones((Nx+1, Ny, Nz), dtype=dtype_) * Da_0,
            hx_z = np.ones((Nx+1, Ny, Nz), dtype=dtype_) * Da_0,
            hy_z = np.ones((Nx, Ny+1, Nz), dtype=dtype_) * Da_0,
            hy_x = np.ones((Nx, Ny+1, Nz), dtype=dtype_) * Da_0,
            hz_x = np.ones((Nx, Ny, Nz+1), dtype=dtype_) * Da_0,
            hz_y = np.ones((Nx, Ny, Nz+1), dtype=dtype_) * Da_0,
        )
        
        self.Db = dict(
            hx_y = np.ones((Nx+1, Ny, Nz), dtype=dtype_) * Db_0,
            hx_z = np.ones((Nx+1, Ny, Nz), dtype=dtype_) * Db_0,
            hy_z = np.ones((Nx, Ny+1, Nz), dtype=dtype_) * Db_0,
            hy_x = np.ones((Nx, Ny+1, Nz), dtype=dtype_) * Db_0,
            hz_x = np.ones((Nx, Ny, Nz+1), dtype=dtype_) * Db_0,
            hz_y = np.ones((Nx, Ny, Nz+1), dtype=dtype_) * Db_0,
        )

    def init_pec(self):
        # initialize the PEC faces
        # this should be called after setting the PML layers

        dx, dy, dz = [conv.m_in(d) for d in self.d_cells]

        # PEC pattern
        for name, pec in self.pec_face.items():
            
            if pec.n_cells > 1 or pec.faces[0] != 4:
                raise ValueError("Only rectangular PEC faces are supported.")
            
            x0, y0, z0 = self.pos_to_idx(np.min(pec.points, axis=0))
            x1, y1, z1 = self.pos_to_idx(np.max(pec.points, axis=0))
    
            x0_c, y0_c, z0_c = self.pos_to_idx(np.min(pec.points, axis=0), mode="cell")
            x1_c, y1_c, z1_c = self.pos_to_idx(np.max(pec.points, axis=0), mode="cell")
    
            # get width of pec in cell units
            idx_d = np.array([x1, y1, z1]) - np.array([x0, y0, z0])
    
            if np.count_nonzero(idx_d) != 2:
                raise ValueError("PEC face must be on cartesian grid, and must be 2D.")
                    
            # PEC is normal to the x-axis, Ez and Ey are parallel to surface
            if (idx_d[0] == 0):
                # ez edges on y axis are inclusive, interior cells on z axis are not
                self.Cb["ez_x"][x0, y0: y1, z0_c: z1_c] = 0
                self.Ca["ez_x"][x0, y0: y1, z0_c: z1_c] = -1
                self.Cb["ez_y"][x0, y0: y1, z0_c: z1_c] = 0
                self.Ca["ez_y"][x0, y0: y1, z0_c: z1_c] = -1
                # ey cells on y axis are not inclusive, edges on z axis are
                self.Cb["ey_z"][x0, y0_c: y1_c, z0: z1 + 1] = 0
                self.Ca["ey_z"][x0, y0_c: y1_c, z0: z1 + 1] = -1
                self.Cb["ey_x"][x0, y0_c: y1_c, z0: z1 + 1] = 0
                self.Ca["ey_x"][x0, y0_c: y1_c, z0: z1 + 1] = -1
            # PEC is normal to the y-axis, Ez and Ex are parallel to surface
            elif (idx_d[1] == 0):
                # ez edges on x axis are inclusive, interior cells on z axis are not
                self.Cb["ez_x"][x0: x1+1, y0, z0_c: z1_c] = 0
                self.Ca["ez_x"][x0: x1+1, y0, z0_c: z1_c] = -1
                self.Cb["ez_y"][x0: x1+1, y0, z0_c: z1_c] = 0
                self.Ca["ez_y"][x0: x1+1, y0, z0_c: z1_c] = -1
                # ex cells on x axis are not inclusive, edges on y axis are
                self.Cb["ex_y"][x0_c: x1_c, y0, z0: z1 + 1] = 0
                self.Ca["ex_y"][x0_c: x1_c, y0, z0: z1 + 1] = -1
                self.Cb["ex_z"][x0_c: x1_c, y0, z0: z1 + 1] = 0
                self.Ca["ex_z"][x0_c: x1_c, y0, z0: z1 + 1] = -1
            # PEC is normal to the z-axis, Ex and Ey are parallel to surface
            elif (idx_d[2] == 0):
                # ey edges on x axis are inclusive, interior cells on y axis are not
                self.Cb["ey_z"][x0: x1+1, y0_c: y1_c, z0] = 0
                self.Ca["ey_z"][x0: x1+1, y0_c: y1_c, z0] = -1
                self.Cb["ey_x"][x0: x1+1, y0_c: y1_c, z0] = 0
                self.Ca["ey_x"][x0: x1+1, y0_c: y1_c, z0] = -1
                # ex cells on x axis are not inclusive, edges on y axis are
                self.Cb["ex_y"][x0_c: x1_c, y0: y1 + 1, z0] = 0
                self.Ca["ex_y"][x0_c: x1_c, y0: y1 + 1, z0] = -1
                self.Cb["ex_z"][x0_c: x1_c, y0: y1 + 1, z0] = 0
                self.Ca["ex_z"][x0_c: x1_c, y0: y1 + 1, z0] = -1

                # self.Db["hz_y"][x0_c: x1_c, y0, z0] = 0
                # self.Db["hz_y"][x0_c: x1_c, y1-1, z0] = 0

                # edge singularity
                d_edge = conv.m_in(0.005)
                # self.Db["hz_y"][x0_c: x1_c, y0-1, z0] *= dy[y0-1] / (dy[y0-1] - d_edge)
                # self.Db["hz_y"][x0_c: x1_c, y1, z0] *= dy[y0-1] / (dy[y0-1] - d_edge)

                # self.Db["hz_y"][x0_c: x1_c, y0-1, z0] *= 
                # self.Db["hz_y"][x0_c: x1_c, y1, z0] *= 1e3

                # self.Cb["ey_x"][x0_c: x1_c, y0-1, z0] *= 2 / (np.log(dy[y0-1] / a))
                # self.Cb["ey_x"][x0_c: x1_c, y1, z0] *= 2 / (np.log(dy[y1] / a))

            else:
                raise ValueError(f"PEC {name} is not 2D")
            
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
                
            # face is normal to the x-axis, Resistor is represented by the Ez components
            if (idx_d[0] == 0):
    
                # width of resistor cell centered around the ez component, ez is on the x/y edges
                dx_r = dx_h[x0-1]
                dy_r = dy_h[y0-1: y1][..., None]
                # z axis is in center of cell
                dz_r = dz[z0: z1][None]
        
                # epsilon of resistor cells
                eps_r = self.eps_ez[x0, y0: y1+1, z0: z1]
        
                # resistance of each cell is spilt so the combined resistance of all cells equals r0
                r_cell = r0 * (eps_r.shape[0] / eps_r.shape[1])
        
                rterm = (r_cell * dx_r * dy_r)
                denom = (eps_r / self.dt) + (dz_r / (2 * rterm))

                ez_idx = tuple([x0, slice(y0, y1+1), slice(z0_c, z1_c)])
                
                self.Ca["ez_x"][ez_idx] = ((eps_r / self.dt) - (dz_r / (2 * rterm))) / denom
                self.Cb["ez_x"][ez_idx] = 1 / (denom)
        
                self.Ca["ez_y"][ez_idx] = ((eps_r / self.dt) - (dz_r / (2 * rterm))) / denom
                self.Cb["ez_y"][ez_idx] = 1 / (denom)

                self.ports[i] = dict(idx=ez_idx, Vs_a=1 / (denom * rterm * eps_r.shape[1]), component="ez")
        
            else:
                raise ValueError(f"Port face is not 2D")

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
        self.Db["hy_x"][h_idx] = (2 * dt) / ((2 * u0 + (sigma_m_hy * dt))) 

        eps_hz = self.eps_hz[h_idx]
        sigma_e_hz = np.broadcast_to(sigma_e_np5, (d_pml,) + self.Da["hz_x"].shape[1:]).copy()
        sigma_e_hz *= (eps_hz / e0)
        sigma_m_hz = sigma_e_hz * u0 / eps_hz

        self.Da["hz_x"][h_idx] = (2 * u0 - (sigma_m_hz * dt)) / (2 * u0 + (sigma_m_hz * dt))
        self.Db["hz_x"][h_idx] = (2 * dt) / ((2 * u0 + (sigma_m_hz * dt)))


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


        coefficients = dict(
            # ex coefficients, edges along y and z do not get updated
            Ca_ex_y = np.array(self.Ca["ex_y"][:, 1:-1, 1:-1], order="C", dtype=dtype_),
            Ca_ex_z = np.array(self.Ca["ex_z"][:, 1:-1, 1:-1], order="C", dtype=dtype_),
            
            Cb_ex_y = np.array(self.Cb["ex_y"][:, 1:-1, 1:-1] * dy_h_inv, order="C", dtype=dtype_),
            Cb_ex_z = np.array(-self.Cb["ex_z"][:, 1:-1, 1:-1] * dz_h_inv, order="C", dtype=dtype_),

            # ey coefficients, edges along x and z do not get updated
            Ca_ey_z = np.array(self.Ca["ey_z"][1:-1, :, 1:-1], order="C", dtype=dtype_),
            Ca_ey_x = np.array(self.Ca["ey_x"][1:-1, :, 1:-1], order="C", dtype=dtype_),
            
            Cb_ey_z = np.array(self.Cb["ey_z"][1:-1, :, 1:-1] * dz_h_inv, order="C", dtype=dtype_),
            Cb_ey_x = np.array(-self.Cb["ey_x"][1:-1, :, 1:-1] * dx_h_inv, order="C", dtype=dtype_),

            # ez coefficients, edges along x and y do not get updated
            Ca_ez_x = np.array(self.Ca["ez_x"][1:-1, 1:-1, :], order="C", dtype=dtype_),
            Ca_ez_y = np.array(self.Ca["ez_y"][1:-1, 1:-1, :], order="C", dtype=dtype_),
            
            Cb_ez_x = np.array(self.Cb["ez_x"][1:-1, 1:-1, :] * dx_h_inv, order="C", dtype=dtype_),
            Cb_ez_y = np.array(-self.Cb["ez_y"][1:-1, 1:-1, :] * dy_h_inv, order="C", dtype=dtype_),

            # hx coefficients
            Da_hx_y = self.Da["hx_y"],
            Da_hx_z = self.Da["hx_z"],
            
            Db_hx_y = -self.Db["hx_y"] * dy_inv,
            Db_hx_z = self.Db["hx_z"] * dz_inv,

            # hy coefficients
            Da_hy_z = self.Da["hy_z"],
            Da_hy_x = self.Da["hy_x"],
            
            Db_hy_z = -self.Db["hy_z"] * dz_inv,
            Db_hy_x = self.Db["hy_x"] * dx_inv,

            # hz coefficients
            Da_hz_x = self.Da["hz_x"],
            Da_hz_y = self.Da["hz_y"],
            
            Db_hz_x = -self.Db["hz_x"] * dx_inv,
            Db_hz_y = self.Db["hz_y"] * dy_inv,
        )


        # field values
        fields = dict(
            ex_y = np.zeros((Nx, Ny+1, Nz+1), dtype=dtype_),
            ex_z = np.zeros((Nx, Ny+1, Nz+1), dtype=dtype_),
            ex = np.zeros((Nx, Ny+1, Nz+1), dtype=dtype_),
            
            ey_z = np.zeros((Nx+1, Ny, Nz+1), dtype=dtype_),
            ey_x = np.zeros((Nx+1, Ny, Nz+1), dtype=dtype_),
            ey = np.zeros((Nx+1, Ny, Nz+1), dtype=dtype_),
            
            ez_x = np.zeros((Nx+1, Ny+1, Nz), dtype=dtype_),
            ez_y = np.zeros((Nx+1, Ny+1, Nz), dtype=dtype_),
            ez = np.zeros((Nx+1, Ny+1, Nz), dtype=dtype_),
            
            hx_y = np.zeros((Nx+1, Ny, Nz), dtype=dtype_),
            hx_z = np.zeros((Nx+1, Ny, Nz), dtype=dtype_),
            hx = np.zeros((Nx+1, Ny, Nz), dtype=dtype_),
            
            hy_z = np.zeros((Nx, Ny+1, Nz), dtype=dtype_),
            hy_x = np.zeros((Nx, Ny+1, Nz), dtype=dtype_),
            hy = np.zeros((Nx, Ny+1, Nz), dtype=dtype_),
            
            hz_x = np.zeros((Nx, Ny, Nz+1), dtype=dtype_),
            hz_y = np.zeros((Nx, Ny, Nz+1), dtype=dtype_),
            hz = np.zeros((Nx, Ny, Nz+1), dtype=dtype_),
        )

        # initialize sources
        sources = []
        for i, p in enumerate(ports):
            port = self.ports[p-1]
            idx, Vs_a, component = port["idx"], port["Vs_a"], port["component"]

            self.ports[p-1]["src"] = v_waveforms[i].copy()

            # convert slice indices to a list of values
            idx_list = [list(np.arange(v.start, v.stop)) if isinstance(v, slice) else [v] for v in idx]

            # create a list of sources for each ez component, with the integer index and scalar waveform data
            Vs_a_flt = Vs_a.flatten()
            for j, idx_j in enumerate(product(*idx_list)):
                values = np.array(-Vs_a_flt[j] * v_waveforms[i], dtype=dtype_, order="C")
                sources.append(
                    dict(values=values, idx=[int(id) for id in idx_j], component=component)
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

        # initialize probes
        probes = []
        for k, p in self.probes.items():
            nx, ny, nz = self.fshape[p["field"]]
            xi, yi, zi = p["index"]

            probes.append(
                dict(
                    values=np.zeros(Nt, dtype=dtype_, order="C"), 
                    field=list(self.fshape.keys()).index(p["field"]),
                    offset=int((xi * ny * nz) + (yi * nz) + zi)
                )
            )

        core_func.solver_run(coefficients, fields, sources, probes, monitors, Nx, Ny, Nz, Nt, n_threads)

        # move monitor values back to the class variable
        for i, (k, m) in enumerate(self.monitors.items()):
            self.monitors[k]["values"] = monitors[i]["values"]

        # get the voltages at each source components
        src_v = [s["values"] for s in sources]
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
            self.probes[k]["values"] = probes[i]["values"]


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

    def get_sparameters(self, frequency, z0=50, source_port=1):
        """
        Returns a column of the sparameter matrix excited by a single port.
        """
        nports = len(self.ports)
        nfrequency = len(frequency)

        # exiting waves (B) from each port
        B = np.zeros((nfrequency, nports), dtype=np.complex128)

        ports = np.arange(1, nports+1)
        z_idx = self.ports[source_port-1]["idx"][2]

        # source port probe, get middle ez components
        src_vp = -np.sum(s.ports[source_port-1]["values"][1] * conv.m_in(self.dz[z_idx])[..., None], axis=0)
        # source port applied voltage
        src_applied = s.ports[source_port-1]["src"]

        # plt.plot(src_applied)
        # plt.plot(src_vp)
        # source port S11
        A = utils.dtft(src_applied, frequency, 1 / s.dt) #* np.exp(-1j * 2 * np.pi * frequency * 3 * s.dt / 2)
        V = utils.dtft(src_vp, frequency, 1 / s.dt)
        
        B[:, source_port-1] = V - (A)

        # exit_ports = np.array([p for p in ports if p != source_port])
        # for p in exit_ports:
        #     name = f"p{p}"

        #     # voltage across port resistor
        #     vp = -s.i_probes[f"{name}_ri"]["values"] * z0
        #     # h-fields are 1/2 time step ahead of the e-fields. Delay current so they are at the same time step
        #     Vp = utils.dtft(vp, s.frequency, 1 / s.dt)
        #     Vp = Vp * np.exp(-1j * 2 * np.pi * s.frequency * s.dt / 2)
        
        #     B[:, p-1] = Vp
        
        return B / A[..., None]
    
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
        cmap="jet"
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
                vmin = 0 if linear else vmax - 50

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
                interpolate_before_map=False,
                render_points_as_spheres=True,
                style="points",
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

        def callback(n_t):

            # n_t = (t * 1e-9) / self.dt
            # n_f = np.clip(int(n_t // n_step), 0, nframe-1)

            self.slider_value = n_t

            for m, f in field_meshes:
                m.point_data["values"][:] = f[int(n_t // n_step)]
          
        self.slider_value = Nt // 2
        self.slider = plotter.add_slider_widget(
            callback,
            [0, Nt],
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
    

        
sbox_h = 0.5
sbox_w = 0.5
sbox_len = 1

sub_h = 0.02
ms_x = (-0.4, 0.5)
ms_y = 0
ms_w = 0.04

substrate = pv.Cube(center=(0, 0, sub_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sub_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

ms_trace = pv.Rectangle([
    (ms_x[0], ms_y - ms_w/2, sub_h),
    (ms_x[0], ms_y + ms_w/2, sub_h),
    (ms_x[1], ms_y + ms_w/2, sub_h)
])

port1_face = pv.Rectangle([
    (ms_x[0], ms_y - ms_w/2, sub_h),
    (ms_x[0], ms_y + ms_w/2, sub_h),
    (ms_x[0], ms_y + ms_w/2, 0),
])

current_face = pv.Rectangle([
    (0, ms_y - ms_w/2, sub_h + 0.001),
    (0, ms_y + ms_w/2, sub_h + 0.001),
    (0, ms_y + ms_w/2, sub_h - 0.001),
])


voltage_line = pv.Line(
    [0, 0, 0], [0, 0, sub_h]
)


# port2_face = pv.Rectangle([
#     (ms_x[1], ms_y - ms_w/2, sub_h),
#     (ms_x[1], ms_y + ms_w/2, sub_h),
#     (ms_x[1], ms_y + ms_w/2, 0),
# ])

s = SolverMesh(sbox, nports=1)
s.add_substrate("sub", substrate, er=3.66, opacity=0.0)
s.add_pec_face("ms1", ms_trace, color="gold")
s.add_lumped_port(1, port1_face)
# s.add_lumped_port(2, port2_face)

s.mesh(d0=[0.02, 0.02, 0.02], n_feature_min=2)
s.init_ports()
s.add_xPML(side="upper")
s.init_pec()
s.add_field_monitor("mon1", "hz", "z", sub_h, 10)
s.add_field_monitor("mon2", "ey", "z", sub_h, 10)
s.add_field_monitor("mon3", "ez", "x", 0, 10)

# 43, 4 for course mesh
# s.add_probe("hz1", "hz", (0, (ms_w / 2) + 0.01, sub_h))
# s.add_probe("ey1", "ey", (0, (ms_w / 2) + 0.01, sub_h))

# s.add_field_monitor("mon2", "ez", "z", 2, 10)
# s.add_field_monitor("mon3", "ez", "x", s.Nx // 2, 10)

self = s

s.add_current_probe("c1", current_face)
s.add_voltage_probe("v1", voltage_line)

plotter = s.render(show_probes=True)
plotter.show()
# print(s.Nx, s.Ny, s.Nz)
# p.camera_position = "xz"
# p.camera.zoom(1.8)

f0 = 10e9
pulse_n = 1200
# width of half pulse in time
t_half = (s.dt * 100)
# center of the pulse in time
t0 = (s.dt * 350)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

ports = 1
v_waveforms = [vsrc]

stime = time.time()
s.run(1, vsrc, n_threads=4)
print(f"Solve Time: {time.time()-stime:.3f}")
print("Grid Cells (k): ", s.Nx * s.Ny * s.Nz / 1e3)



Db_0 = s.dt / u0
Cb_0 = s.dt / e0 
p = s.plot_cooeficients("ex_z", "a", "z", sub_h, point_size=15, cmap="brg", normalization=None)
p.show()

# p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet")
# p.show(title="EM Solver")

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)


sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v = self.vi_probe_values("v1")

plt.plot(line_v)

IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt)
ZP = VP / IP

fig, ax = plt.subplots()
rfn.plots.draw_smithchart(ax)
plt.plot(S11.real, S11.imag)


fig, ax = plt.subplots()
ax.plot(frequency/1e9, rfn.conv.z_gamma(S11))
ax.plot(frequency/1e9, ZP.real)
plt.show()


# fig, ax = plt.subplots()
# plt.plot(s.probes["c1_1"]["values"])
# mplm.axis_marker(y = np.max(s.probes["c1_1"]["values"]))
# plt.show()



# 
# Nx, Ny, Nz = 126, 26, 26
