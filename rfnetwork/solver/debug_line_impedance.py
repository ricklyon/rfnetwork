import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time

from IPython.display import Image as ipyimage
import rfnetwork as rfn
import mpl_markers as mplm
import matplotlib.colors as mcolors

u0 = const.u0
e0 = const.e0
c0 = const.c0

import sys

sys.argv = sys.argv[0:1]


class Solver_SingleLayer():

    def __init__(
        self, 
        frequency: np.ndarray,
        pattern: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
        eps_z: float,
        ms_z: float
    ):
        self.frequency = frequency
        self.dx = dx
        self.dy = dy

        # z-index of substrate layer
        sub_z = slice(0, ms_z)

        # number of total cells in the x, y, z direction
        Nx = pattern.shape[0]
        Ny = pattern.shape[1]
        Nz = len(dz)

        # cell sizes in z direction
        self.dz = dz

        # compute maximum time step that ensures convergence, use freespace propagation speed as worst case
        length_min = np.array([np.min(dx), np.min(dy), np.min(self.dz)])
        dmin = 1 / np.sqrt(((1 / length_min)**2).sum())
        dt = 0.95 * (dmin / const.c0)

        #################
        # Materials
        #################

        sig_0 = 0
        Ca_0 = (2 * e0 - (sig_0 * dt)) / (2 * e0 + (sig_0 * dt))
        Cb_0 = (2 * dt) / ((2 * e0 + (sig_0 * dt)))
        # Da = (2 * u0 - (sigma_m * dt)) / (2 * u0 + (sigma_m * dt))
        Db_0 = (dt) / (u0)

        # coefficient in front of the previous time values of E
        Ca_ex = np.ones((Nx, Ny+1, Nz+1)) * Ca_0
        Ca_ey = np.ones((Nx+1, Ny, Nz+1)) * Ca_0
        Ca_ez = np.ones((Nx+1, Ny+1, Nz)) * Ca_0

        # coefficient in front of the difference terms of H
        Cb_ex = np.ones((Nx, Ny+1, Nz+1)) * Cb_0
        Cb_ey = np.ones((Nx+1, Ny, Nz+1)) * Cb_0
        Cb_ez = np.ones((Nx+1, Ny+1, Nz)) * Cb_0

        # substrate
        eps_exy = eps_z.copy()
        eps_exy = np.concatenate([eps_exy, [e0]])
        Cb_ez[..., :] = (2 * dt) / ((2 * eps_z + (sig_0 * dt)))
        Cb_ex[..., :] = (2 * dt) / ((2 * eps_exy + (sig_0 * dt)))
        Cb_ey[..., :] = (2 * dt) / ((2 * eps_exy + (sig_0 * dt)))

        # ex and ey are on the boundary between material cells, compute the average of the substrate
        # FIX for non-uniform grids along z
        # eps_exy = (2 * eps_z[:-1] * eps_z[1:]) / (eps_z[:-1] + eps_z[1:])
        
        # eps_exy = np.sqrt((eps_z[:-1] * eps_z[1:]))
        # eps_exy = ((eps_z[:-1] + eps_z[1:])) / 2
        
        
        # eps_exy = np.concatenate([eps_exy, [e0]])
        # eps_exy = np.ones_like(eps_exy) * e0

        # eps_avg = (e0 + (e0 * 3.66)) / 2
        # eps_exy[ms_z] = eps_avg
        # Cb_ex[..., :] = (2 * dt) / ((2 * eps_exy + (sig_0 * dt))) 
        # Cb_ey[..., :] = (2 * dt) / ((2 * eps_exy + (sig_0 * dt))) 
        # Cb_ex[..., ms_z] = (Cb_ex[..., ms_z - 1] + Cb_ex[..., ms_z + 1]) / 2
        # Cb_ey[..., ms_z] = (Cb_ey[..., ms_z - 1] + Cb_ey[..., ms_z + 1]) / 2

        # PEC pattern
        # conductivity of PEC
        sig_pec = 1e15

        # set ex coefficient to PEC if either cell next to it (along y) is set as copper
        ex_patt = (pattern[:, :-1] | pattern[:, 1:]) 
        
        Ca_ms = (2 * eps_exy[ms_z] - (sig_pec * dt)) / (2 * eps_exy[ms_z] + (sig_pec * dt))
        Cb_ms = (2 * dt) / ((2 * eps_exy[ms_z] + (sig_pec * dt)))

        Ca_ex[..., 1:-1, ms_z] = np.where(
            ex_patt, Ca_ms, Ca_ex[..., 1:-1, ms_z]
        )
        Cb_ex[..., 1:-1, ms_z] = np.where(
            ex_patt, Cb_ms, Cb_ex[..., 1:-1, ms_z]
        )

        # set ey coefficient to PEC if either cell next to it (along x) is set as copper
        ey_patt = (pattern[:-1] | pattern[1:])

        Ca_ey[1:-1, :, ms_z] = np.where(
            ey_patt, Ca_ms, Ca_ey[1:-1, :, ms_z]
        )
        Cb_ey[1:-1, :, ms_z] = np.where(
            ey_patt, Cb_ms, Cb_ey[1:-1, :, ms_z]
        )

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
            hx_y = np.ones((Nx+1, Ny, Nz)),
            hx_z = np.ones((Nx+1, Ny, Nz)),
            hy_z = np.ones((Nx, Ny+1, Nz)),
            hy_x = np.ones((Nx, Ny+1, Nz)),
            hz_x = np.ones((Nx, Ny, Nz+1)),
            hz_y = np.ones((Nx, Ny, Nz+1)),
        )
        
        self.Db = dict(
            hx_y = np.ones((Nx+1, Ny, Nz)) * Db_0,
            hx_z = np.ones((Nx+1, Ny, Nz)) * Db_0,
            hy_z = np.ones((Nx, Ny+1, Nz)) * Db_0,
            hy_x = np.ones((Nx, Ny+1, Nz)) * Db_0,
            hz_x = np.ones((Nx, Ny, Nz+1)) * Db_0,
            hz_y = np.ones((Nx, Ny, Nz+1)) * Db_0,
        )

        self.ms_z = ms_z
        self.dt = dt
        self.pattern = pattern
        self.ports = dict()
        self.v_probes = dict()
        self.i_probes = dict()
        self.fields = dict()
        self.eps_exy = eps_exy
        self.eps_ez = eps_z
        self.v_src = None
        
        # field shapes
        self.fshape = dict(
            ex=(Nx, Ny+1, Nz+1),
            ey=(Nx+1, Ny, Nz+1),
            ez=(Nx+1, Ny+1, Nz),
            hx=(Nx+1, Ny, Nz),
            hy=(Nx, Ny+1, Nz),
            hz=(Nx, Ny, Nz+1)
        )

        # grid edge cell locations in meters
        gx = np.concatenate([[0], np.cumsum(dx)])
        gy = np.concatenate([[0], np.cumsum(dy)])
        gz = np.concatenate([[0], np.cumsum(dz)])

        # middle cell locations
        gx_h = (gx[1:] + gx[:-1]) / 2
        gy_h = (gy[1:] + gy[:-1]) / 2
        gz_h = (gz[1:] + gz[:-1]) / 2

        # locations of all field components in grid
        self.floc = dict(
            ex=(gx_h, gy, gz),
            ey=(gx, gy_h, gz),
            ez=(gx, gy, gz_h),
            hx=(gx, gy_h, gz_h),
            hy=(gx_h, gy, gz_h),
            hz=(gx_h, gy_h, gz)
        )

        # locations of the centers of each cell
        self.cell_loc = (gx_h, gy_h, gz_h)
        # location of grid edges
        self.edge_loc = (gx, gy, gz)
    

    def add_v_probe(self, name, x, y):
        """
        Add a voltage probe at the Ez location of (x, y)
        """

        self.v_probes[name] = dict(x=x, y=y)

    def add_i_probe(self, name, x, y, axis=None):
        """
        Add a current probe. Current flowing inside the cells given at (x, y) is measured, one of which must be a slice.
        Slice endpoint is inclusive.
        """
        # get the axis the current is oriented along
        if axis is None:
            if isinstance(x, slice):
                axis = "y"
            elif isinstance(y, slice):
                axis = "x"
            else:
                raise ValueError("Either x or y of a current probe must be a slice if axis is not specified.")
                
        self.i_probes[name] = dict(x=x, y=y, axis=axis)

    def add_field_monitor(self, name: str, field: str, axis: str, position: int, t_step: int):
        """
        Add field monitor along a slice through the 3D volume

        Parameters
        ----------
        name : str
        field : {'ex', 'ey', 'ez', 'hx', 'hy', 'hz'}
        axis : {'x', 'y', 'z'}
        position : int
            index along axis of the slice
        t_step : int, default: 1
            number of time steps between each capture.
        """

        if field not in self.fshape.keys():
            raise ValueError(f"Unsuported field: {field}. Expecting one of: {tuple(self.fshape.keys())}")
        if axis not in ('x', 'y', 'z'):
            raise ValueError(f"Unsupported axis: {axis}")

        # get the spatial shape of the field slice
        axis_i = dict(x=0, y=1, z=2)[axis]
        shape = list(self.fshape[field])
        axis_len = shape.pop(axis_i)

        if (position < 0) or (position > self.fshape[field][axis_i] - 1):
            raise ValueError(f"Position out of bounds, expecting value from 0 - {axis_len}")

        slc = [slice(None)] * 3
        slc[axis_i] = position
        
        self.fields[name] = dict(
            field=field, axis=axis, position=position, t_step=t_step, slc=tuple(slc), shape=tuple(shape)
        )

        
    def add_port(self, number, direction, ez_x, ez_y, r0=50, ref_plane=3):

        # resistive load
        r_x = ez_x
        r_y = ez_y
        r_z = slice(0, self.ms_z)
        name = f"p{number}"

        dx_r = (self.dx[ez_x - 1] + self.dx[ez_x]) / 2
        dy_r = (self.dy[ez_y - 1] + self.dy[ez_y]) / 2
        dz_r = self.dz[r_z]

        eps_port = self.eps_ez[r_z]

        # distribute resistive load amoung cells along z
        if r0 is not None:
            rterm = ((r0 / self.ms_z) * dx_r * dy_r)
            denom = (eps_port / self.dt) + (dz_r / (2 * rterm))
    
            self.Ca["ez_x"][r_x, r_y, r_z] = ((eps_port / self.dt) - (dz_r / (2 * rterm))) / denom
            self.Cb["ez_x"][r_x, r_y, r_z] = 1 / (denom)
    
            self.Ca["ez_y"][r_x, r_y, r_z] = ((eps_port / self.dt) - (dz_r / (2 * rterm))) / denom
            self.Cb["ez_y"][r_x, r_y, r_z] = 1 / (denom)
        else:
            rterm = ((50 / self.ms_z) * dx_r * dy_r)
            denom = (eps_port / self.dt) + (dz_r / (2 * rterm))

        # add voltage probe
        if direction[0] == "x":
            
            xv = r_x + ref_plane if direction[1] == "+" else r_x - ref_plane
            yv = r_y
            # current cells
            yi_1 = r_y - np.argmin(self.pattern[xv, r_y-1::-1])
            yi_2 = r_y + np.argmin(self.pattern[xv, r_y:]) - 1

            self.add_i_probe(name + "_i1", xv - 1, slice(yi_1, yi_2))
            self.add_i_probe(name + "_i2", xv, slice(yi_1, yi_2))
            self.add_v_probe(name + "_v", xv, yv)

            # voltage probe across resistor
            self.add_v_probe(name + "_rv", r_x, r_y)
            self.add_i_probe(name + "_ri", r_x, r_y, axis="z")

            # a = conv.m_in(0.001)
            # if direction[1] == "+":
            #     self.Db["hy_x"][r_x - 1, r_y, r_z] = (2 * self.dt) / (u0 * np.log(self.dx[r_x - 1] / a))
            # else:
            #     self.Db["hy_x"][r_x, r_y, r_z] = (2 * self.dt) / (u0 * np.log(self.dx[r_x] / a))

        # elif direction == "y+":
        #     xv, yv = r_x, r_y + ref_plane
        # elif direction == "y-":
        #     xv, yv = r_x, r_y - ref_plane

        # u0_c1 = u0 * 0.9

        # self.Db["hy_x"][r_x - 1, r_y, r_z] = self.dt / u0_c1
        # self.Db["hy_x"][r_x, r_y, r_z] = self.dt / u0_c1
        # self.Db["hx_y"][r_x, r_y - 1, r_z] = self.dt / u0_c1
        # self.Db["hx_y"][r_x, r_y, r_z] = self.dt / u0_c1
            
        # add to port list
        self.ports[name] = dict(
            x=r_x, y=r_y, z=r_z, Vs_a=1 / (denom * rterm * self.ms_z), direction=direction
        )


    def run(self, port, v_waveform):

        Nt = len(v_waveform)
        self.v_src = v_waveform

        # numpy type for the field values
        dtype_ = np.float32
        dx, dy, dz = self.dx, self.dy, self.dz
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

        # half cell lengths between h components
        dx_h = (dx[1:] + dx[:-1]) / 2
        dy_h = (dy[1:] + dy[:-1]) / 2
        dz_h = (dz[1:] + dz[:-1]) / 2

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

        src_x, src_y, src_z, Vs_a, _ = self.ports[port].values()

        # initialize probes
        for k, p in self.v_probes.items():
            self.v_probes[k]["values"] = np.zeros(Nt, dtype=dtype_)

        for k, p in self.i_probes.items():
            self.i_probes[k]["values"] = np.zeros(Nt, dtype=dtype_)

        # initialize field monitors
        for k, f in self.fields.items():
            self.fields[k]["values"] = np.zeros(((Nt // f["t_step"])+1,) + f["shape"], dtype=dtype_)

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
            
            ez_x[src_x, src_y, src_z] -= (Vs_a) * (v_waveform[n])
            ez_y[src_x, src_y, src_z] -= (Vs_a) * (v_waveform[n])
            ez = ez_x + ez_y

            # ez_x[src_x, src_y, 0] -= (Vs_a) * (v_waveform[n])
            # ez_y[src_x, src_y, 0] -= (Vs_a) * (v_waveform[n])
            # ez = ez_x + ez_y
            
            # ey_z[10, 10:12, 3] = (1e6 * v_waveform[n+1])
            # ey_z[1, 30:32, 30]  -= (Cb_ey_z[30, 30:32, 30] * dz[0]) * (1e6 * v_waveform[n])
            # ey_x[30, 30:32, 30]  += (Cb_ey_x[30, 30:32, 30] * dx[0]) * (1e4 * v_waveform[n])
            # ey = ey_z + ey_x

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

            # update voltage probes
            for name, p in self.v_probes.items():
                self.v_probes[name]["values"][n+1] = -np.sum(ez[p["x"], p["y"], :self.ms_z] * dz[:self.ms_z])
            
            # update current probe
            for name, p in self.i_probes.items():
                x, y, axis = p["x"], p["y"], p["axis"]
                ms_z = self.ms_z

                # cells along long axis circulating the current
                if axis == "x":
                    # hy components below and above trace. For current in the +x direction, the h components below the trace are
                    # positive, above components are negative.
                    hy_p = np.array([1, -1])[None] * hy[x, y.start: y.stop + 2, ms_z - 1: ms_z + 1] * dy_h[y.start - 1: y.stop + 1][..., None]
                    # hz components on the sides of the trace.
                    hz_p = (-hz[x, y.start - 1, ms_z] + hz[x, y.stop + 1, ms_z]) * dz_h[ms_z - 1]
                    self.i_probes[name]["values"][n + 1] = np.sum(hy_p) + hz_p

                elif axis == "z":
                     
                    # sample fields in the center of the substrate along z. If the middle falls on a integer cell edge,
                    # average the two fields above and below it
                    z_mid = self.ms_z / 2
                    zp = slice(int(z_mid - 1), int(z_mid + 1)) if z_mid % 1 == 0 else int(np.floor(z_mid))

                    ix = (np.mean(hx[x, y-1, zp]) - np.mean(hx[x, y, zp])) * dx_h[x - 1]
                    iy = (-np.mean(hy[x-1, y, zp]) + np.mean(hy[x, y, zp])) * dy_h[y - 1]

                    self.i_probes[name]["values"][n + 1] = ix + iy
    
                # else:  # axis == "y"
                #     # hx components below and above the trace. For current the +y direction, the h components below the trace are
                #     # negative, above are positive
                #     hx_p = np.array([-1, 1])[None] * hx[x.start: x.stop + 2, y, ms_z - 1: ms_z + 1] * dx_h[x.start - 1: x.stop + 1][..., None]
                #     # hz components on the sides of the trace
                #     hz_p = (hz[x.start - 1, y, ms_z] - hz[x.stop + 1, y, ms_z]) * dz_h[ms_z - 1]
                #     self.i_probes[name]["values"][n + 1] = np.sum(hx_p) + hz_p

            # update field monitors
            for name, f in self.fields.items():
                t_step = f["t_step"]
                if (n % t_step) != 0:
                    continue
                    
                field = dict(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz)[f["field"]]
                f["values"][n // t_step] = field[f["slc"]]
                
                    
    def add_yPML(self, d_pml=10, side="upper"):
        """
        Add PML layer to the top face of the solution box.
        """
        m_pml = 3 # sigma profile order
        
        dt = self.dt
        dy = self.dy[-1]
        eta0 = np.sqrt(u0 / e0)
        # now define the values of sigma and sigma_m from the profiles
        sigma_max = 0.8 * (m_pml + 1) / (eta0 * dy)
    
        # define sigma profile in the PML region on the right side of the grid. 
        i_pml = np.arange(0, d_pml)[None, :, None]
    
        # sigma on the cell edges. Components on the edge of the PML have a sigma of 0.
        sigma_e_n = sigma_max * ((i_pml) / (d_pml))**m_pml
        # sigma in the middle of the cells. First Hz component in the PML is 0.5 cells into the PML
        sigma_e_np5 = sigma_max * ((i_pml + 0.5) / (d_pml))**m_pml

        # magnetic conductivity
        # plt.figure()
        # plt.plot(np.arange(0, d_pml, 1), sigma_e_n.squeeze())
        # plt.plot(np.arange(0.5, d_pml + .5, 1), sigma_e_np5.squeeze())

        e_idx = slice(d_pml, 0, -1) if side=="lower" else slice(-d_pml-1, -1)
        h_idx = slice(d_pml-1, None, -1) if side=="lower" else slice(-d_pml, None)

        # ez
        # first ez component is at the edge of the PML where sigma = 0, last component is at the solve boundary and not updated
        # sigma / eps must be constant across y and z, page 291 in taflove
        # scale sigma by eps so that sigma / eps is constant
        eps_ez = self.eps_ez
        sigma_ez = np.broadcast_to(sigma_e_n, (self.Ca["ez_y"].shape[0], d_pml, self.Ca["ez_y"].shape[-1])).copy()
        sigma_ez *= (eps_ez / e0)

        self.Ca["ez_y"][:, e_idx] = (2 * eps_ez - (sigma_ez * dt)) / (2 * eps_ez + (sigma_ez * dt))
        self.Cb["ez_y"][:, e_idx] = (2 * dt) / ((2 * eps_ez + (sigma_ez * dt)))
        # (sigma_ez / eps_ez)[5, 0, :]

        # ex
        eps_ex = self.eps_exy
        sigma_ex = np.broadcast_to(sigma_e_n, (self.Ca["ex_y"].shape[0], d_pml, self.Ca["ex_y"].shape[-1])).copy()
        sigma_ex *= (eps_ex / e0)
        # print((sigma_ey / eps_ey)[5, 0, :])

        # exclude the ey components in the trace from the PML
        ex_patt = (pattern[:, :-1] | pattern[:, 1:])[:, h_idx]

        self.Ca["ex_y"][:, e_idx] = np.where(
            ex_patt[..., None],
            self.Ca["ex_y"][:, e_idx],
            (2 * eps_ex - (sigma_ex * dt)) / (2 * eps_ex + (sigma_ex * dt)),
        )
        self.Cb["ex_y"][:, e_idx] = np.where(
            ex_patt[..., None],
            self.Cb["ex_y"][:, e_idx],
            (2 * dt) / ((2 * eps_ex + (sigma_ex * dt))),
        )

        # hx/hy components are in the middle of the PML cells, use half cell indices
        eps_hx = self.eps_ez
        simga_e_hx = np.broadcast_to(sigma_e_np5, (self.Da["hx_y"].shape[0], d_pml, self.Da["hx_y"].shape[-1])).copy()
        simga_e_hx *= (eps_hx / e0)
        sigma_m_hx = simga_e_hx * u0 / eps_hx

        self.Da["hx_y"][:, h_idx] = (2 * u0 - (sigma_m_hx * dt)) / (2 * u0 + (sigma_m_hx * dt))
        self.Db["hx_y"][:, h_idx] = (2 * dt) / ((2 * u0 + (sigma_m_hx * dt)))

        eps_hz = self.eps_exy
        sigma_e_hz = np.broadcast_to(sigma_e_np5, (self.Da["hz_y"].shape[0], d_pml, self.Da["hz_y"].shape[-1])).copy()
        sigma_e_hz *= (eps_hz / e0)
        sigma_m_hz = sigma_e_hz * u0 / eps_hz

        self.Da["hz_y"][:, h_idx] = (2 * u0 - (sigma_m_hz * dt)) / (2 * u0 + (sigma_m_hz * dt))
        self.Db["hz_y"][:, h_idx] = (2 * dt) / ((2 * u0 + (sigma_m_hz * dt)))

    def add_xPML(self, d_pml=10, side="upper"):
        """
        Add PML layer to the top face of the solution box.
        """
        m_pml = 3 # sigma profile order

        dt = self.dt
        dx = self.dx[-1]
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

        e_idx = slice(d_pml, 0, -1) if side=="lower" else slice(-d_pml-1, -1)
        h_idx = slice(d_pml-1, None, -1) if side=="lower" else slice(-d_pml, None)

        # ez
        # first ez component is at the edge of the PML where sigma = 0, last component is at the solve boundary and not updated
        # sigma / eps must be constant across y and z, page 291 in taflove
        # scale sigma by eps so that sigma / eps is constant
        eps_ez = self.eps_ez
        sigma_ez = np.broadcast_to(sigma_e_n, (d_pml,) + self.Ca["ez_x"].shape[1:]).copy()
        sigma_ez *= (eps_ez / e0)
        
        self.Ca["ez_x"][e_idx] = (2 * eps_ez - (sigma_ez * dt)) / (2 * eps_ez + (sigma_ez * dt))
        self.Cb["ez_x"][e_idx] = (2 * dt) / ((2 * eps_ez + (sigma_ez * dt))) 
        # (sigma_ez / eps_ez)[5, 0, :]

        # ey
        eps_ey = self.eps_exy
        sigma_ey = np.broadcast_to(sigma_e_n, (d_pml,) + self.Ca["ey_x"].shape[1:]).copy()
        sigma_ey *= (eps_ey / e0)
        # print((sigma_ey / eps_ey)[5, 0, :])

        # exclude the ey components in the trace from the PML
        ey_patt = (pattern[:-1] | pattern[1:])[h_idx]

        self.Ca["ey_x"][e_idx] = np.where(
            ey_patt[..., None], 
            self.Ca["ey_x"][e_idx],
            (2 * eps_ey - (sigma_ey * dt)) / (2 * eps_ey + (sigma_ey * dt)),
        )
        self.Cb["ey_x"][e_idx] = np.where(
            ey_patt[..., None], 
            self.Cb["ey_x"][e_idx],
            (2 * dt) / ((2 * eps_ey + (sigma_ey * dt))),
        )

        # hx/hy components are in the middle of the PML cells, use half cell indices
        eps_hy = self.eps_ez
        simga_e_hy = np.broadcast_to(sigma_e_np5, (d_pml,) + self.Da["hy_x"].shape[1:]).copy()
        simga_e_hy *= (eps_hy / e0)
        sigma_m_hy = simga_e_hy * u0 / eps_hy

        self.Da["hy_x"][h_idx] = (2 * u0 - (sigma_m_hy * dt)) / (2 * u0 + (sigma_m_hy * dt))
        self.Db["hy_x"][h_idx] = (2 * dt) / ((2 * u0 + (sigma_m_hy * dt))) 

        eps_hz = self.eps_exy
        sigma_e_hz = np.broadcast_to(sigma_e_np5, (d_pml,) + self.Da["hz_x"].shape[1:]).copy()
        sigma_e_hz *= (eps_hz / e0)
        sigma_m_hz = sigma_e_hz * u0 / eps_hz


        self.Da["hz_x"][h_idx] = (2 * u0 - (sigma_m_hz * dt)) / (2 * u0 + (sigma_m_hz * dt))
        self.Db["hz_x"][h_idx] = (2 * dt) / ((2 * u0 + (sigma_m_hz * dt))) 


    def add_zPML(self, d_pml=10):
        """
        Add PML layer to the top face of the solution box.
        """
        m_pml = 3 # sigma profile order

        dt = self.dt
        dz = self.dz[-1]
        eta0 = np.sqrt(u0 / e0)
        # now define the values of sigma and sigma_m from the profiles
        sigma_max = 0.8 * (m_pml + 1) / (eta0 * dz)

        # define sigma profile in the PML region on the right side of the grid. 
        i_pml = np.arange(0, d_pml)[None, None]

        # sigma on the cell edges. Components on the edge of the PML have a sigma of 0.
        sigma_e_n = sigma_max * ((i_pml) / (d_pml))**m_pml
        # sigma in the middle of the cells. First Hz component in the PML is 0.5 cells into the PML
        sigma_e_np5 = sigma_max * ((i_pml + 0.5) / (d_pml))**m_pml
        sigma_m_np5 = sigma_e_np5 * u0 / e0

        # ex
        self.Ca["ex_z"][..., -d_pml-1:-1] = (2 * e0 - (sigma_e_n * dt)) / (2 * e0 + (sigma_e_n * dt))
        self.Cb["ex_z"][..., -d_pml-1:-1] = (2 * dt) / ((2 * e0 + (sigma_e_n * dt))) 

        # ey
        self.Ca["ey_z"][..., -d_pml-1:-1] = (2 * e0 - (sigma_e_n * dt)) / (2 * e0 + (sigma_e_n * dt))
        self.Cb["ey_z"][..., -d_pml-1:-1] = (2 * dt) / ((2 * e0 + (sigma_e_n * dt))) 

        # hx/hy components are in the middle of the PML cells, use half cell indices
        self.Da["hx_z"][..., -d_pml:] = (2 * u0 - (sigma_m_np5 * dt)) / (2 * u0 + (sigma_m_np5 * dt))
        self.Db["hx_z"][..., -d_pml:] = (2 * dt) / ((2 * u0 + (sigma_m_np5 * dt))) 

        self.Da["hy_z"][..., -d_pml:] = (2 * u0 - (sigma_m_np5 * dt)) / (2 * u0 + (sigma_m_np5 * dt))
        self.Db["hy_z"][..., -d_pml:] = (2 * dt) / ((2 * u0 + (sigma_m_np5 * dt))) 


    def render(self) -> pv.Plotter:
        """
        Plot the model geometry
        """

        ms_z = self.ms_z
        gx, gy, gz = [conv.in_m(f) for f in self.edge_loc]
        gx_h, gy_h, gz_h = [conv.in_m(f) for f in self.cell_loc]
        
        grid = pv.RectilinearGrid(gx, gy, gz)
        

        plotter = pv.Plotter()
        # plotter.enable_parallel_projection()
        # add grid
        plotter.add_mesh(grid, style="wireframe", line_width=0.05, color="k", opacity=0.05)
        
        # add copper features
        pattern_g = pv.RectilinearGrid(gx, gy, gz[ms_z: ms_z+1])
        pattern_g.cell_data["values"] = self.pattern.flatten(order="F")

        # Create the colormap from the list of colors
        cmap_two_colors = mcolors.LinearSegmentedColormap.from_list(
            "custom_cmap", ["none", "gold"]
        )

        plotter.add_mesh(
            pattern_g,
            opacity=1,
            cmap=cmap_two_colors,
            show_scalar_bar=False
        )
        plotter.add_axes()
        
        # substrate
        Gx, Gy, Gz = gx[-1], gy[-1], gz[ms_z]
        sub = pv.Cube(
            center=(Gx / 2, Gy / 2, Gz/2),
            x_length=Gx,
            y_length=Gy,
            z_length=Gz
        )
        plotter.add_mesh(sub, opacity=0.1, color="green")

        # show voltage probes
        for name, p in self.v_probes.items():
            x, y = gx[p["x"]], gy[p["y"]]
            vline = pv.Line((x, y, 0), (x, y, gz[ms_z]))
            plotter.add_mesh(vline, line_width=2, color="k", opacity=0.4)

        # show current probes
        for name, p in self.i_probes.items():
            x, y, axis = p["x"], p["y"], p["axis"]
            gx_p = gx_h[x]

            if axis == "x":
                rect = pv.Rectangle([
                    (gx_p, gy_h[y.start-1], gz_h[self.ms_z - 1]),
                    (gx_p, gy_h[y.stop+1], gz_h[self.ms_z - 1]),
                    (gx_p, gy_h[y.stop+1], gz_h[self.ms_z])
                ])

                plotter.add_mesh(rect, line_width=2, color="k", opacity=0.4)
                    

        plotter.show_grid(n_zlabels=2, font_size=9)
        
        return plotter

    def animate_fields(
        self, filename, monitor_name: list, view="xz", vmax=30, vmin=-20, zoom=1.3, el=10, az=0, opacity="linear"
    ):

        monitor_name = np.atleast_1d(monitor_name)
        plotter = self.render()
        field_meshes = []

        for i, m_name in enumerate(monitor_name):

            monitor = self.fields[m_name]
            field = monitor["values"]
            slc = monitor["slc"]
            t_step = monitor["t_step"]

            field = np.where(np.abs(field) < 1e-12, 1e-12, field)
            field_db = np.clip(20 * np.log10(np.abs(field)), vmin, vmax)
        
            nframe = len(field)

            floc_m = self.floc[monitor["field"]]
            floc_in = [conv.in_m(f) for f in floc_m]
            # slice each axis of grid locations, forcing integer indices into slices
            g = [floc_in[i][s] if isinstance(s, slice) else floc_in[i][s: s+1] for i, s in enumerate(slc)]
            
            fmesh = pv.RectilinearGrid(*g)
            fmesh.point_data['values'] = field_db[0].flatten(order="F")
            
            plotter.add_mesh(fmesh, cmap="jet", scalars="values", clim=[vmin, vmax], show_scalar_bar=False, opacity=opacity[i])

            field_meshes += [(fmesh, field_db)]

        plotter.add_axes()
        plotter.camera_position = view
        plotter.camera.elevation += el
        plotter.camera.azimuth += az
        plotter.camera.zoom(zoom)
        
        plotter.add_scalar_bar(
            title="Ez [dB]\n", vertical=False, label_font_size=11, title_font_size=14
        )
        
        plotter.open_gif(filename)

        for n in range(nframe):
            for m, f in field_meshes:
                m.point_data["values"][:] = f[n].flatten(order="F")
                
            plotter.add_title(f"t={n * t_step * self.dt * 1e9:.2f}ns")
            # plotter.render()
            plotter.write_frame()

        # Closes and finalizes movie
        plotter.close()

            
    def get_sparameters(self, z0=50, source_port=1):
        """
        Returns a column of the sparameter matrix excited by a single port.
        """
        nports = len(self.ports)
        nfrequency = len(self.frequency)

        # exiting waves (B) from each port
        B = np.zeros((nfrequency, nports), dtype=np.complex128)

        ports = np.arange(1, nports+1)
        exit_ports = np.array([p for p in ports if p != source_port])

        # source port S11
        name = f"p{source_port}"

        src_vp = s.v_probes[f"{name}_rv"]["values"]
        src_vp_i = s.i_probes[f"{name}_ri"]["values"] * z0
        
        A = utils.dtft(self.v_src, s.frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * s.frequency * s.dt)
        V = utils.dtft(src_vp, s.frequency, 1 / s.dt)
        Vi = utils.dtft(src_vp_i, s.frequency, 1 / s.dt)
        
        B[:, source_port-1] = V - (A)

        for p in exit_ports:
            name = f"p{p}"

            # voltage across port resistor
            vp = -s.i_probes[f"{name}_ri"]["values"] * z0
            # h-fields are 1/2 time step ahead of the e-fields. Delay current so they are at the same time step
            Vp = utils.dtft(vp, s.frequency, 1 / s.dt)
            # Vp = Vp * np.exp(-1j * 2 * np.pi * s.frequency * s.dt / 2)
        
            B[:, p-1] = Vp
        
        return B / A[..., None]
            

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

# CTRL + SHIFT + C
# command pallet

f0 = 10e9
Nz = 25

ms_z = 2
eps_z = np.ones(Nz) * e0
eps_z[:ms_z] = 3.66 * e0


Nx = 120
Ny = 30

p1_x, p1_y = 10, Ny//2

p2_x, p2_y = Nx-10, Ny//2

pattern = np.zeros((Nx, Ny), dtype=np.int32)
pattern[p1_x:p2_x, p1_y-1:p1_y+1] = 1

w = 0.04
d = 0.02
dx0 = conv.m_in(0.02)
dy0 = conv.m_in(0.02)
dz0 = conv.m_in(d/ms_z)

dx = np.ones(Nx) * dx0
dy = np.ones(Ny) * dy0
dz = np.ones(Nz) * dz0

# line appears to be slightly too low in impedance, I think because the edge of PEC is approximated by the Yee grid,
# rather than make the mesh drastically smaller, compensate the line width.
# This appears to be independent of dz or dx, and only a function of dy adjacent to the trace.
dy[p1_y-1:p1_y+1] = conv.m_in(w/2 - (w/10))
dy[p1_y-2] = conv.m_in(w/2)
dy[p1_y+1] = conv.m_in(w/2)
dy[p1_y-3] = (conv.m_in((w/2)) + dy0) / 2
dy[p1_y+2] = (conv.m_in((w/2)) + dy0) / 2

s = Solver_SingleLayer(frequency, pattern, dx, dy, dz, eps_z, ms_z=ms_z)

s.add_port(1, "x+", p1_x, p1_y, r0=50, ref_plane=5)
s.add_port(2, "x-", p2_x, p2_y, r0=50, ref_plane=5)

s.add_field_monitor("ez_ms", "ez", "z", ms_z, 15)
s.add_field_monitor("ez_xz", "ez", "y", p1_y, 15)
s.add_field_monitor("ez_yz", "ez", "x", 50, 15)

# s.add_xPML(d_pml=10, side="lower")
# s.add_xPML(d_pml=10, side="upper")
s.add_yPML(d_pml=5, side="lower")
s.add_yPML(d_pml=5, side="upper")
s.add_zPML(d_pml=5)

pulse_n = 1500
# width of half pulse in time
t_half = 4e-11  # (dt * (pulse_n // 8))
# center of the pulse in time
t0 = (s.dt * 340)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
# i_amp = 0.5e-3
# j_amp = i_amp / (dx[p1_x] * dy[p1_y])
vsrc = 1e-2 * (np.sin(2*np.pi*f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32).squeeze()

Ca_0 = 1
Cb_0 = (s.dt) / (e0)
Cb_e = (s.dt) / (e0 * 3.66)
Db_0 = (s.dt) / (u0)

sig = 1e7
dt = s.dt
Ca = (2 * e0 - (sig * dt)) / (2 * e0 + (sig * dt))
Cb = (2 * dt) / ((2 * e0 + (sig * dt)))
er_eff = 2.5
Cb_ms = (dt) / (e0 * 1.5)

s.run("p1", vsrc)

S = s.get_sparameters()


S11 = S[:, 0]
S21 = S[:, 1]


msline50 = rfn.elements.MSLine(
    w=w,
    h=0.02,
    er=3.66,
)

zref = msline50.get_properties(10e9).sel(value="z0")[0]

fig, ax = plt.subplots()
ax.plot(frequency, conv.z_gamma(S11).real)
ax.set_ylim([0, 120])
mplm.line_marker(x=10e9)
mplm.axis_marker(y=zref)

fig, ax = plt.subplots()
rfn.plots.draw_smithchart(ax)
plt.plot(S11.real, S11.imag)

fig, ax = plt.subplots()
ref_s11 = msline50(conv.in_m((Nx - 20) * dx0)).evaluate(frequency)["s"].sel(b=1, a=1)
ax.plot(frequency/1e9, conv.db20_lin(S11))
ax.plot(frequency/1e9, conv.db20_lin(ref_s11))
ax.set_ylim([-50, 0])

fig, ax = plt.subplots()
ax.plot(frequency/1e9, conv.db20_lin(S21))

pv.set_jupyter_backend('trame')
p = s.render()
p.show()

p = s.animate_fields("msline_2.gif", ["ez_ms", "ez_yz"], el=30, zoom=1.4, vmin=-20, vmax=20, az=40, view="xz", opacity=[0.8, 1])
ipyimage(filename='msline_2.gif')





