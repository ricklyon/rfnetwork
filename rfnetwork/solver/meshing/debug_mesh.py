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
            idx += [np.argmax(diff >= -1e-6) if diff[-1] > 0 else len(g)]

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
        


    def mesh(self):

        s.create_grid()
        
        dx, dy, dz = self.d_cells
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
                dx_r = self.dx_h[x0-1]
                dy_r = self.dy_h[y0-1: y1][..., None]
                # z axis is in center of cell
                dz_r = self.dz[z0: z1][None]
        
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

        field, direction = component[:2], component[3]
        
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

s.mesh()
s.setup_ports()

# s.render().show()

p = s.plot_cooeficients("ez_x", "a", "x", -1.0, point_size=15, cmap="brg", vmin=-1)
p.camera_position = "xz"
p.camera.zoom(1.8)
p.show_grid(font_size=10, grid=True, use_3d_text=False, location="default")
p.show()

self = s





