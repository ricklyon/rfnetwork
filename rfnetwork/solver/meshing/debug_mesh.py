import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time

from IPython.display import Image as ipyimage
import rfnetwork as rfn
import mpl_markers as mplm
import matplotlib.colors as mcolors

import sys
import matplotlib
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

    def point_to_idx(self, p, mode="edge"):

        idx = []
        mode = [mode] * 3 if isinstance(mode, str) else mode
        
        grid = [self.g_edges[i] if m == "edge" else self.g_cells[i] for i, m in enumerate(mode)]
        for i, g in enumerate(grid):
            diff = (g - p[i])
            # if no cell is above the point, return the length of the axis, otherwise return the first cell that is larger than the point.
            idx += [np.argmax(diff >= -1e-3) if diff[-1] > 0 else len(g)]

        return tuple(idx)
    
    def create_grid(self, d0=0.02):
        edges = [np.array([], dtype=np.float32) for i in range(3)]
        objects = [self.bounding_box] + list(self.pec_face.values()) + [sub[0] for sub in self.substrate.values()]
        
        for obj in objects:
            # round points to minimum precision supported by the mesh
            p_edges = np.around(obj.points.T, decimals=3).astype(np.float32)
        
            for i in range(3):
                edges[i] = np.unique(np.concatenate([edges[i], p_edges[i]]))

        # list of cell widths for each axis
        cell_d = [[], [], []]
        # iterate over x, y, z axis
        for i in range(3):
            edges_i = edges[i]
        
            # list of distances between each edge
            d_e = np.diff(edges_i)
            
            for d in d_e:
                # how many cells of the default size will fit in this interval
                nx = d / d0
            
                # use a minimum of two cells between adjacent features
                if nx < 2:
                    cell_d[i] += [d / 2] * 2
            
                else:
                    # make the cell a bit smaller than the default to account for the remainder of nx
                    nx = int(nx + 1) if nx % 1 > 1e-3 else int(nx)
                    # d0 * nx will be larger than d if there was a remainder on nx, divide up this difference amoung each cell
                    d0_e = d0 - (((d0 * nx) - d) / nx)
                    cell_d[i] += [d0_e] * nx
            
        # TODO: blend large differences in adjacent cell widths
        
        gx, gy, gz = [np.around(np.concatenate([[self.sbox_min[i]], self.sbox_min[i] + np.cumsum(cell_d[i])]), decimals=6) for i in range(3)]
        
        self.grid_mesh = pv.RectilinearGrid(gx, gy, gz)
        # cell widths
        dx, dy, dz = np.diff(gx), np.diff(gy), np.diff(gz)
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
            x0, y0, z0 = self.point_to_idx(np.min(sub.points, axis=0), mode="cell")
            x1, y1, z1 = self.point_to_idx(np.max(sub.points, axis=0), mode="cell")
        
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
        


    def mesh(self, d0):

        s.create_grid(d0)
        
        dx, dy, dz = self.d_cells
        Nx, Ny, Nz = self.n_cells
        
        # compute maximum time step that ensures convergence, use freespace propagation speed as worst case
        length_min = np.array([np.min(dx), np.min(dy), np.min(dz)])
        dmin = 1 / np.sqrt(((1 / length_min)**2).sum())
        dt = 0.95 * (dmin / const.c0_in)
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

        self.eps_ex = np.ones(self.fshape["ex"]) * e0
        self.eps_ey = np.ones(self.fshape["ey"]) * e0
        self.eps_ez = np.ones(self.fshape["ez"]) * e0

        # ey component is on the y and z edge of the cell
        self.eps_ex[:, 1:-1, 1:-1] = ((eps_y[..., :-1] * (dz0/2) + eps_y[..., 1:] * (dz1/2)) / (dz0/2 + dz1/2))
        # ey component is on the x and z edge of the cell
        self.eps_ey[1:-1, :, 1:-1] = ((eps_x[..., :-1] * (dz0/2) + eps_x[..., 1:] * (dz1/2)) / (dz0/2 + dz1/2))
        # ez component is on the x and y edge of the cell, average eps from adjacent cells on both axis
        self.eps_ez[1:-1, 1:-1] = ((eps_x[:, :-1] * (dy0/2) + eps_x[:, 1:] * (dy1/2)) / (dy0/2 + dy1/2))

        # coefficient in front of the previous time values of E
        Ca_ex = np.ones((Nx, Ny+1, Nz+1)) * Ca_0
        Ca_ey = np.ones((Nx+1, Ny, Nz+1)) * Ca_0
        Ca_ez = np.ones((Nx+1, Ny+1, Nz)) * Ca_0
        
        # coefficient in front of the difference terms of H
        Cb_ex = dt / self.eps_ex
        Cb_ey = dt / self.eps_ey
        Cb_ez = dt / self.eps_ez

        # PEC pattern
        for name, pec in self.pec_face.items():
            
            if pec.n_cells > 1 or pec.faces[0] != 4:
                raise ValueError("Only rectangular PEC faces are supported.")
            
            x0, y0, z0 = self.point_to_idx(np.min(pec.points, axis=0))
            x1, y1, z1 = self.point_to_idx(np.max(pec.points, axis=0))
    
            x0_c, y0_c, z0_c = self.point_to_idx(np.min(pec.points, axis=0), mode="cell")
            x1_c, y1_c, z1_c = self.point_to_idx(np.max(pec.points, axis=0), mode="cell")
    
            # get width of pec in cell units
            idx_d = np.array([x1, y1, z1]) - np.array([x0, y0, z0])
    
            if np.count_nonzero(idx_d) != 2:
                raise ValueError("PEC face must be on cartesian grid, and must be 2D.")
                    
            # PEC is normal to the x-axis, Ez and Ey are parallel to surface
            if (idx_d[0] == 0):
                # ez edges on y axis are inclusive, interior cells on z axis are not
                Cb_ez[x0, y0: y1, z0_c: z1_c] = 0
                Ca_ez[x0, y0: y1, z0_c: z1_c] = -1
                # ey cells on y axis are not inclusive, edges on z axis are
                Cb_ey[x0, y0_c: y1_c, z0: z1 + 1] = 0
                Ca_ey[x0, y0_c: y1_c, z0: z1 + 1] = -1
            # PEC is normal to the y-axis, Ez and Ex are parallel to surface
            elif (idx_d[1] == 0):
                # ez edges on x axis are inclusive, interior cells on z axis are not
                Cb_ez[x0: x1+1, y0, z0_c: z1_c] = 0
                Ca_ez[x0: x1+1, y0, z0_c: z1_c] = -1
                # ex cells on x axis are not inclusive, edges on y axis are
                Cb_ex[x0_c: x1_c, y0, z0: z1 + 1] = 0
                Ca_ex[x0_c: x1_c, y0, z0: z1 + 1] = -1
            # PEC is normal to the z-axis, Ex and Ey are parallel to surface
            elif (idx_d[2] == 0):
                # ey edges on x axis are inclusive, interior cells on y axis are not
                Cb_ey[x0: x1+1, y0_c: y1_c, z0] = 0
                Ca_ey[x0: x1+1, y0_c: y1_c, z0] = -1
                # ex cells on x axis are not inclusive, edges on y axis are
                Cb_ex[x0_c: x1_c, y0: y1 + 1, z0] = 0
                Ca_ex[x0_c: x1_c, y0: y1 + 1, z0] = -1
            else:
                raise ValueError(f"PEC {name} is not 2D")
    
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
            hx_y = np.ones((Nx+1, Ny, Nz)) * Da_0,
            hx_z = np.ones((Nx+1, Ny, Nz)) * Da_0,
            hy_z = np.ones((Nx, Ny+1, Nz)) * Da_0,
            hy_x = np.ones((Nx, Ny+1, Nz)) * Da_0,
            hz_x = np.ones((Nx, Ny, Nz+1)) * Da_0,
            hz_y = np.ones((Nx, Ny, Nz+1)) * Da_0,
        )
        
        self.Db = dict(
            hx_y = np.ones((Nx+1, Ny, Nz)) * Db_0,
            hx_z = np.ones((Nx+1, Ny, Nz)) * Db_0,
            hy_z = np.ones((Nx, Ny+1, Nz)) * Db_0,
            hy_x = np.ones((Nx, Ny+1, Nz)) * Db_0,
            hz_x = np.ones((Nx, Ny, Nz+1)) * Db_0,
            hz_y = np.ones((Nx, Ny, Nz+1)) * Db_0,
        )

    def setup_ports(self):
        r0 = 50
        face = self.port_face[0]

        self.ports = [None] * len(self.port_face)
        
        dx, dy, dz = [conv.m_in(d) for d in self.d_cells]
        dx_h, dy_h, dz_h = [conv.m_in(d) for d in self.dh_cells]
        
        for i, face in enumerate(self.port_face):
    
            if face.n_cells > 1 or face.faces[0] != 4:
                raise ValueError("Only rectangular port faces are supported.")
            
            x0, y0, z0 = self.point_to_idx(np.min(face.points, axis=0))
            x1, y1, z1 = self.point_to_idx(np.max(face.points, axis=0))
    
            x0_c, y0_c, z0_c = self.point_to_idx(np.min(face.points, axis=0), mode="cell")
            x1_c, y1_c, z1_c = self.point_to_idx(np.max(face.points, axis=0), mode="cell")
    
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
        
                # resistance of each cell is spilt so the combined reistance of all cells equals r0
                r_cell = r0 * (eps_r.shape[0] / eps_r.shape[1])
        
                rterm = (r_cell * dx_r * dy_r)
                denom = (eps_r / self.dt) + (dz_r / (2 * rterm))

                ez_idx = tuple([x0, slice(y0, y1+1), slice(z0_c, z1_c)])
                
                self.Ca["ez_x"][ez_idx] = ((eps_r / self.dt) - (dz_r / (2 * rterm))) / denom
                self.Cb["ez_x"][ez_idx] = 1 / (denom)
        
                self.Ca["ez_y"][ez_idx] = ((eps_r / self.dt) - (dz_r / (2 * rterm))) / denom
                self.Cb["ez_y"][ez_idx] = 1 / (denom)

                self.ports[i] = dict(idx=ez_idx, Vs_a=1 / (denom * rterm * eps_r.shape[1]))
        
            else:
                raise ValueError(f"Port face is not 2D")

    
    def run(self, ports, v_waveforms):

        if isinstance(ports, int):
            ports = [ports]

        v_waveforms = np.atleast_2d(v_waveforms)
            
        Nt = len(v_waveforms[0])
        self.v_src = v_waveforms

        # numpy type for the field values
        dtype_ = np.float32
        dx, dy, dz = [conv.m_in(d) for d in self.d_cells]
        dx_h, dy_h, dz_h = [conv.m_in(d) for d in self.dh_cells]
        Nx, Ny, Nz = len(dx), len(dy), len(dz)

        # field values
        ex_y = np.zeros((Nx, Ny+1, Nz+1), dtype=dtype_)
        ex_z = np.zeros((Nx, Ny+1, Nz+1), dtype=dtype_)
        
        ey_z = np.zeros((Nx+1, Ny, Nz+1), dtype=dtype_)
        ey_x = np.zeros((Nx+1, Ny, Nz+1), dtype=dtype_)
        
        ez_x = np.zeros((Nx+1, Ny+1, Nz), dtype=dtype_)
        ez_y = np.zeros((Nx+1, Ny+1, Nz), dtype=dtype_)
        
        hx_y = np.zeros((Nx+1, Ny, Nz), dtype=dtype_)
        hx_z = np.zeros((Nx+1, Ny, Nz), dtype=dtype_)
        
        hy_z = np.zeros((Nx, Ny+1, Nz), dtype=dtype_)
        hy_x = np.zeros((Nx, Ny+1, Nz), dtype=dtype_)
        
        hz_x = np.zeros((Nx, Ny, Nz+1), dtype=dtype_)
        hz_y = np.zeros((Nx, Ny, Nz+1), dtype=dtype_)

        dx_inv = 1 / dx[:, None, None]
        dy_inv = 1 / dy[None, :, None]
        dz_inv = 1 / dz[None, None, :]

        dx_h_inv = 1 / dx_h[:, None, None]
        dy_h_inv = 1 / dy_h[None, :, None]
        dz_h_inv = 1 / dz_h[None, None, :]
        
        # ex coefficients, edges along y and z do not get updated
        Ca_ex_y = self.Ca["ex_y"][:, 1:-1, 1:-1]
        Ca_ex_z = self.Ca["ex_z"][:, 1:-1, 1:-1]
        
        Cb_ex_y = self.Cb["ex_y"][:, 1:-1, 1:-1] * dy_h_inv
        Cb_ex_z = -self.Cb["ex_z"][:, 1:-1, 1:-1] * dz_h_inv

        # ey coefficients, edges along x and z do not get updated
        Ca_ey_z = self.Ca["ey_z"][1:-1, :, 1:-1]
        Ca_ey_x = self.Ca["ey_x"][1:-1, :, 1:-1]
        
        Cb_ey_z = self.Cb["ey_z"][1:-1, :, 1:-1] * dz_h_inv
        Cb_ey_x = -self.Cb["ey_x"][1:-1, :, 1:-1] * dx_h_inv

        # ez coefficients, edges along x and y do not get updated
        Ca_ez_x = self.Ca["ez_x"][1:-1, 1:-1, :]
        Ca_ez_y = self.Ca["ez_y"][1:-1, 1:-1, :]
        
        Cb_ez_x = self.Cb["ez_x"][1:-1, 1:-1, :] * dx_h_inv
        Cb_ez_y = -self.Cb["ez_y"][1:-1, 1:-1, :] * dy_h_inv

        # hx coefficients
        Da_hx_y = self.Da["hx_y"]
        Da_hx_z = self.Da["hx_z"]
        
        Db_hx_y = -self.Db["hx_y"] * dy_inv
        Db_hx_z = self.Db["hx_z"] * dz_inv

        # hy coefficients
        Da_hy_z = self.Da["hy_z"]
        Da_hy_x = self.Da["hy_x"]
        
        Db_hy_z = -self.Db["hy_z"] * dz_inv
        Db_hy_x = self.Db["hy_x"] * dx_inv

        # hz coefficients
        Da_hz_x = self.Da["hz_x"]
        Da_hz_y = self.Da["hz_y"]
        
        Db_hz_x = -self.Db["hz_x"] * dx_inv
        Db_hz_y = self.Db["hz_y"] * dy_inv

        # initialize port voltages
        for i, p in enumerate(self.ports):
            self.ports[i]["v_probe"] = np.zeros(Nt, dtype=dtype_)

        for i, p in enumerate(ports):
            self.ports[p-1]["src"] = v_waveforms[i].copy()

        hx = hx_y
        hy = hy_z
        hz = hz_x
        
        # loop over each time step
        for n in range(Nt - 1):

            # grid starts at bottom left corner. A half-cell is placed in front of each component so all field values have 
            # the same number of values. The extra half cell components are not updated.

            ###########
            # ex update
            # edges along y and z do not get updated
            ex_yd = Cb_ex_y * np.diff(hz, axis=1)[:, :, 1:-1]
            ex_zd = Cb_ex_z * np.diff(hy, axis=2)[:, 1:-1, :]

            # in PML
            ex_y[:, 1:-1, 1:-1] = (Ca_ex_y * ex_y[:, 1:-1, 1:-1]) + ex_yd
            ex_z[:, 1:-1, 1:-1] = (Ca_ex_z * ex_z[:, 1:-1, 1:-1]) + ex_zd
            ex = ex_y + ex_z
            # normal region
            # ex = Ca_ex * ex[:, 1:-1, 1:-1] + (ex_zd + ex_yd)

            # ex[:, 1:-1, 1:-1] = (Ca_ex * ex[:, 1:-1, 1:-1]) + Cb_ex * (
            #     (np.diff(hz, axis=1) * dy_h_inv)[:, :, 1:-1] - 
            #     (np.diff(hy, axis=2) * dz_h_inv)[:, 1:-1, :]
            # )

            ###########
            # ey update
            # edges along x and z do not get updated
            ey_zd = Cb_ey_z * np.diff(hx, axis=2)[1:-1, :, :]
            ey_xd = Cb_ey_x * np.diff(hz, axis=0)[:, :, 1:-1]

            # in PML
            ey_z[1:-1, :, 1:-1] = (Ca_ey_z * ey_z[1:-1, :, 1:-1]) + ey_zd
            ey_x[1:-1, :, 1:-1] = (Ca_ey_x * ey_x[1:-1, :, 1:-1]) + ey_xd
            ey = ey_z + ey_x
            # normal region
            # ey = Ca_ey * ey[:, 1:-1, 1:-1] + (ey_zd + ey_xd)
            
            # ey[1:-1, :, 1:-1] = (Ca_ey * ey[1:-1, :, 1:-1]) + Cb_ey * (
            #     (np.diff(hx, axis=2) * dz_h_inv)[1:-1, :, :] - 
            #     (np.diff(hz, axis=0) * dx_h_inv)[:, :, 1:-1]
            # )

            ###########
            # ez update
            # edges along x and y do not get updated
            ez_xd = Cb_ez_x * np.diff(hy, axis=0)[:, 1:-1, :]
            ez_yd = Cb_ez_y * np.diff(hx, axis=1)[1:-1, :, :]

            # in PML
            ez_x[1:-1, 1:-1, :] = (Ca_ez_x * ez_x[1:-1, 1:-1, :]) + ez_xd
            ez_y[1:-1, 1:-1, :] = (Ca_ez_y * ez_y[1:-1, 1:-1, :]) + ez_yd
            ez = ez_x + ez_y
            # normal region
            # ez = Ca_ez * ez[:, 1:-1, 1:-1] + (ez_xd + ez_yd)
            
            # ez[1:-1, 1:-1, :] = (Ca_ez * ez[1:-1, 1:-1, :]) + Cb_ez * (
            #     (np.diff(hy, axis=0) * dx_h_inv)[:, 1:-1, :] - 
            #     (np.diff(hx, axis=1) * dy_h_inv)[1:-1, :, :]
            # )

            ############
            # add resistive voltage source
            # Vs_a is already divided by sub_z, so we don't need to split the voltage among each ez component

            for i, p in enumerate(ports):
                p_idx, Vs_a, v_probe, v_src = self.ports[p-1].values()

                ez_x[p_idx] -= (Vs_a) * (v_src[n]) #+ v_src[n + 1]) / 2
                ez_y[p_idx] -= (Vs_a) * (v_src[n]) # + v_src[n + 1]) / 2

                ez = ez_x + ez_y

            # update port probes
            for i, p in enumerate(self.ports):
                
                # update port voltage values for ports normal to x
                p_x, p_y, p_z = p["idx"]

                # sum voltage along z axis
                v_port = -np.sum(ez[p_idx] * dz[p_z][None], axis=-1)
                # if port spans three ez components along y, use the center voltage, otherwise average the two endpoints
                # to approximate the voltage in the center
                port_len = len(v_port)
                if len(v_port) % 2 != 0:
                    self.ports[i]["v_probe"][n+1] = v_port[port_len // 2]
                else:
                    self.ports[i]["v_probe"][n+1] = np.mean(v_port[(port_len // 2) - 1: (port_len // 2) + 1])
                

            ###########
            # hx update
            hx_yd = Db_hx_y * np.diff(ez, axis=1)
            hx_zd = Db_hx_z * np.diff(ey, axis=2)

            # in PML
            hx_y = Da_hx_y * hx_y + hx_yd
            hx_z = Da_hx_z * hx_z + hx_zd
            hx = hx_y + hx_z
            # normal region
            # hx +=  (hx_y + hx_z)
            
            # hx = (hx) + (
            #     self.Db["hx_ey"] * (np.diff(ey, axis=2) * dz_inv) - 
            #     self.Db["hx_ez"] * (np.diff(ez, axis=1) * dy_inv)
            # )

            ###########
            # hy update
            hy_zd = Db_hy_z * np.diff(ex, axis=2)
            hy_xd = Db_hy_x * np.diff(ez, axis=0)

            # in PML
            hy_z = Da_hy_z * hy_z + hy_zd
            hy_x = Da_hy_x * hy_x + hy_xd
            hy = hy_z + hy_x
            # normal region
            # hy +=  (hy_z + hy_x)
            
            # hy = (hy) + (
            #     self.Db["hy_ez"]  * (np.diff(ez, axis=0) * dx_inv) - 
            #     self.Db["hy_ex"]  * (np.diff(ex, axis=2) * dz_inv)
            # )

            ###########
            # hz update
            hz_xd = Db_hz_x * np.diff(ey, axis=0) 
            hz_yd = Db_hz_y * np.diff(ex, axis=1)

            # in PML
            hz_x = Da_hz_x * hz_x + hz_xd
            hz_y = Da_hz_y * hz_y + hz_yd
            hz = hz_x + hz_y
            # normal region
            # hz += (hz_x + hz_y)
            
            # hz = (hz) + (
            #     self.Db["hz_ex"]  * (np.diff(ex, axis=1) * dy_inv) - 
            #     self.Db["hz_ey"]  * (np.diff(ey, axis=0) * dx_inv)
            # )
                
            
        
    def render(self) -> pv.Plotter:
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

        plotter.show_grid(font_size=9)

        return plotter

    def plot_cooeficients(self, component, value, axis, pos, vmin=None, vmax=None, opacity=1, cmap="jet", point_size=10, normalization=None):
            
        plotter = self.render()

        idx = [slice(None)] * 3
        axis_i = dict(x=0, y=1, z=2)[axis]

        pos_full = [0] * 3
        pos_full[axis_i] = pos

        if value == "a":
            values = self.Ca[component] if component[0] == "e" else self.Da[component]
        else:
            values = self.Cb[component] if component[0] == "e" else self.Db[component]

        if normalization:
            values = values / normalization
            
        if vmax is None:
            vmax = np.max(values)
        if vmin is None:
            vmin = np.min(values)

        # if axis length of values is larger than the number of cells, it is a edge component, get edge index
        mode = "edge" if values.shape[axis_i] > self.n_cells[axis_i] else "cell"
        idx[axis_i] = self.point_to_idx(pos_full, mode=mode)[axis_i]

        field = component[:2]
        
        floc = self.floc[field]

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
            title=f"{component}, {value}\n", vertical=False, label_font_size=11, title_font_size=14
        )
        
        return plotter

    def get_sparameters(self, frequency, z0=50, source_port=1):
        """
        Returns a column of the sparameter matrix excited by a single port.
        """
        nports = len(self.ports)
        nfrequency = len(frequency)

        # exiting waves (B) from each port
        B = np.zeros((nfrequency, nports), dtype=np.complex128)

        ports = np.arange(1, nports+1)

        # source port probe
        src_vp = s.ports[source_port-1]["v_probe"]
        # source port applied voltage
        src_applied = s.ports[source_port-1]["src"]

        # plt.plot(src_applied)
        # plt.plot(src_vp)
        # source port S11
        A = utils.dtft(src_applied, frequency, 1 / s.dt) #* np.exp(-1j * 2 * np.pi * frequency * 3 * s.dt / 2)
        V = utils.dtft(src_vp, frequency, 1 / s.dt)
        
        B[:, source_port-1] = V - (A)

        exit_ports = np.array([p for p in ports if p != source_port])
        # for p in exit_ports:
        #     name = f"p{p}"

        #     # voltage across port resistor
        #     vp = -s.i_probes[f"{name}_ri"]["values"] * z0
        #     # h-fields are 1/2 time step ahead of the e-fields. Delay current so they are at the same time step
        #     Vp = utils.dtft(vp, s.frequency, 1 / s.dt)
        #     Vp = Vp * np.exp(-1j * 2 * np.pi * s.frequency * s.dt / 2)
        
        #     B[:, p-1] = Vp
        
        return B / A[..., None]
        
sbox_h = 0.5
sbox_w = 0.5
sbox_len = 2.5

sub_h = 0.02
ms_x = (-1, 1)
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

port2_face = pv.Rectangle([
    (ms_x[1], ms_y - ms_w/2, sub_h),
    (ms_x[1], ms_y + ms_w/2, sub_h),
    (ms_x[1], ms_y + ms_w/2, 0),
])

s = SolverMesh(sbox, nports=2)
s.add_substrate("sub", substrate, er=3.66, opacity=0.0)
s.add_pec_face("ms1", ms_trace, color="gold")
s.add_lumped_port(1, port1_face)
s.add_lumped_port(2, port2_face)

s.mesh(d0=0.010)
s.setup_ports()

s.render().show()
print(s.Nx, s.Ny, s.Nz)

# p = s.plot_cooeficients("ez_x", "a", "y", 0, point_size=15, cmap="brg", vmin=-1)
# p.show()
# p.camera_position = "xz"
# p.camera.zoom(1.8)
# p.show_grid(font_size=10, grid=True, use_3d_text=False, location="default")
# p.show()

self = s

f0 = 10e9
pulse_n = 1800
# width of half pulse in time
t_half = (s.dt * 100)
# center of the pulse in time
t0 = (s.dt * 300)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

ports = 1
v_waveforms = [vsrc]

s.run(1, vsrc)

plt.plot(vsrc)
plt.plot(s.ports[0]["v_probe"])

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

sdata = s.get_sparameters(frequency)
S11 = sdata[:, 0]

fig, ax = plt.subplots()
ax.plot(frequency/1e9, conv.db20_lin(S11))

fig, ax = plt.subplots()
rfn.plots.draw_smithchart(ax)
plt.plot(S11.real, S11.imag)
plt.show()

