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

class Dummy():
    pass

self = Dummy()

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
        print(sub_z)

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
        Cb_ez[..., :] = (2 * dt) / ((2 * eps_z + (sig_0 * dt))) 
        # Cb_ex[..., 1:] = (2 * dt) / ((2 * eps_z + (sig_0 * dt))) 
        # Cb_ey[..., 1:] = (2 * dt) / ((2 * eps_z + (sig_0 * dt))) 

        # ex and ey are on the boundary between material cells, compute the average of the substrate
        # FIX for non-uniform grids along z
        eps_exy = (2 * eps_z[:-1] * eps_z[1:]) / (eps_z[:-1] + eps_z[1:])
        eps_exy = np.concatenate([[e0], eps_exy, [e0]])

        # eps_avg = (e0 + (e0 * er))/ 2
        Cb_ex[..., :] = (2 * dt) / ((2 * eps_exy + (sig_0 * dt))) 
        Cb_ey[..., :] = (2 * dt) / ((2 * eps_exy + (sig_0 * dt))) 
        # Cb_ex[..., ms_z] = (Cb_ex[..., ms_z - 1] + Cb_ex[..., ms_z + 1]) / 2
        # Cb_ey[..., ms_z] = (Cb_ey[..., ms_z - 1] + Cb_ey[..., ms_z + 1]) / 2

        # PEC pattern
        # conductivity of PEC
        sig_pec = 1e7

        # set ex coefficient to PEC if either cell next to it (along y) is set as copper
        ex_patt = (pattern[:, :-1] | pattern[:, 1:]) 
        ex_sig = ex_patt * sig_pec

        Ca_ex[..., 1:-1, ms_z] = np.where(
            ex_patt, (2 * eps_exy[ms_z] - (ex_sig * dt)) / (2 * eps_exy[ms_z] + (ex_sig * dt)), Ca_ex[..., 1:-1, ms_z]
        )
        Cb_ex[..., 1:-1, ms_z] = np.where(
            ex_patt, (2 * dt) / ((2 * eps_exy[ms_z] + (ex_sig * dt))), Cb_ex[..., 1:-1, ms_z]
        )

        # set ey coefficient to PEC if either cell next to it (along x) is set as copper
        ey_patt = (pattern[:-1] | pattern[1:]) 
        ey_sig = ey_patt * sig_pec
        
        Ca_ey[1:-1, :, ms_z] = np.where(
            ey_patt, (2 * eps_exy[ms_z] - (ey_sig * dt)) / (2 * eps_exy[ms_z] + (ey_sig * dt)), Ca_ey[1:-1, :, ms_z]
        )
        Cb_ey[1:-1, :, ms_z] = np.where(
            ey_patt, (2 * dt) / ((2 * eps_exy[ms_z] + (ey_sig * dt))) , Cb_ey[1:-1, :, ms_z]
        )

        self.Ca = dict(
            ex_y = Ca_ex,
            ex_z = Ca_ex,
            ey_z = Ca_ey,
            ey_x = Ca_ey,
            ez_x = Ca_ez,
            ez_y = Ca_ez
        )
    
        self.Cb = dict(
            ex_y = Cb_ex,
            ex_z = Cb_ex,
            ey_z = Cb_ey,
            ey_x = Cb_ey,
            ez_x = Cb_ez,
            ez_y = Cb_ez
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
        self.c_probes = dict()
        self.eps_exy = eps_exy
        self.eps_ez = eps_z

    def add_v_probe(self, name, x, y):

        self.v_probes[name] = dict(x=x, y=y)

    def add_c_probe(self, name, x, y1, y2):

        self.c_probes[name] = dict(x=x, y1=y1, y2=y2)

    def run(self, port, v_waveform, gif_step=10):

        Nt = len(v_waveform)

        # numpy type for the field values
        dtype_ = np.float32
        dx, dy, dz = self.dx, self.dy, self.dz
        Nx, Ny, Nz = len(dx), len(dy), len(dz)
        
        self.gif_step = gif_step
        self.gif_fields = np.zeros(((Nt // self.gif_step) + 1, Nx+1, Ny+1, Nz), dtype=dtype_)

        
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

        src_x, src_y, src_z, Vs_a = self.ports[port].values()

        # field values at ports
        port_fields = dict()
        for k, p in self.ports.items():
            port_fields[k] = dict(
                ez = np.zeros(Nt, dtype=dtype_),
                hx = np.zeros((Nt, 2), dtype=dtype_),
                hy = np.zeros((Nt, 2), dtype=dtype_),
            )

        for k, p in self.v_probes.items():
            self.v_probes[k]["values"] = np.zeros(Nt, dtype=dtype_)

        for k, p in self.c_probes.items():
            self.c_probes[k]["values"] = np.zeros(Nt, dtype=dtype_)

        hx = hx_y
        hy = hy_z
        hz = hz_x
        stime = time.time()
        # loop over each time step
        for n in range(Nt -  1):

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
            
            ez_x[src_x, src_y, src_z] -= (Vs_a * (v_waveform[n + 1]) / 2)
            ez_y[src_x, src_y, src_z] -= (Vs_a * (v_waveform[n + 1]) / 2)
            ez[src_x, src_y, src_z] = ez_x[src_x, src_y, src_z] + ez_y[src_x, src_y, src_z]
            
            # # ey_z[10, 10:12, 3] = (1e6 * v_waveform[n+1])
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

            # update port values
            for k, p in self.ports.items():
                x, y, z, _ = p.values()
                port_fields[k]["ez"][n + 1] = -np.sum(ez[x, y, :self.ms_z] * dz[:self.ms_z])
                port_fields[k]["hx"][n + 1] = hx[x, y-1: y+1, self.ms_z-1]
                port_fields[k]["hy"][n + 1] = hy[x-1: x+1, y, self.ms_z-1]

            if (n % self.gif_step) == 0:
                self.gif_fields[n // self.gif_step] = ez

            for k, p in self.v_probes.items():
                self.v_probes[k]["values"][n+1] = -np.sum(ez[p["x"], p["y"], :self.ms_z] * dz[:self.ms_z])
            
            # current probe
            for k, p in self.c_probes.items():
                x, y1, y2 = p["x"], p["y1"], p["y2"]
                self.c_probes[k]["values"][n+1] = (
                    np.sum((hy[x, y1:y2+2, self.ms_z-1] - hy[x, y1:y2+2, self.ms_z]) * dy_h[y1-1:y2+1]) + (-hz[x, y1-1, 1] + hz[x, y2+1, self.ms_z]) * dz_h[self.ms_z-1]
                )
                # print(conv.in_m(dy_h[y1-1:y2+1]))

        return port_fields

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
        sigma_ez = np.broadcast_to(sigma_e_n, (self.Ca["ez_x"].shape[0], d_pml, self.Ca["ez_x"].shape[-1])).copy()
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

        # s = np.arange(15)
        # s[slice(d_pml-1, None, -1) ]
        # s[10::-1]
        # s[slice(d_pml+1, 0, -1)]

        # s[slice(-d_pml-1, -1)]

        # s[slice(d_pml, None, -1)]
        # s[-d_pml:]

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

    

    def add_port(self, name, ez_x, ez_y, r0=50, ind_comp = 0.75):
        # resistive load
        r_x = ez_x

        r_y = ez_y
        r_z = self.ms_z - 1

        dx_r = (self.dx[ez_x - 1] + self.dx[ez_x]) / 2
        dy_r = (self.dy[ez_y - 1] + self.dy[ez_y]) / 2
        dz_r = self.dz[r_z]

        # print(dx_r, dy_r, dz_r)

        print(conv.in_m(dx_r), conv.in_m(dy_r), conv.in_m(dz_r))

        rterm = (r0 * dx_r * dy_r)

        eps_port = self.eps_ez[r_z]
        denom = (eps_port / self.dt) + (dz_r / (2 * rterm))

        # ez component for resistor
        self.Ca["ez_x"][r_x, r_y, r_z] = ((eps_port / self.dt) - (dz_r / (2 * rterm))) / denom
        self.Cb["ez_x"][r_x, r_y, r_z] = 1 / (denom)

        self.Ca["ez_y"][r_x, r_y, r_z] = ((eps_port / self.dt) - (dz_r / (2 * rterm))) / denom
        self.Cb["ez_y"][r_x, r_y, r_z] = 1 / (denom)

        # TODO: add wires for multi-cell 
        # 0 ohm resistors to connect the lumped element to the trace
        self.Ca["ez_x"][r_x, r_y, :r_z] = -1
        self.Cb["ez_x"][r_x, r_y, :r_z] = 0

        self.Ca["ez_y"][r_x, r_y, :r_z] = -1
        self.Cb["ez_y"][r_x, r_y, :r_z] = 0
        ###

        # h components around load, thin wire model
        a = conv.m_in(1e-9)
        u0_c1 = u0 * ind_comp

        # # shorten load along the x direction
        # # port faces +x
        if not self.pattern[r_x - 1, r_y] and self.pattern[r_x, r_y]:
            print("+x")
            self.Db["hy_x"][r_x - 1, r_y, :r_z+1] = (2 * self.dt) / (u0_c1 * np.log(self.dx[r_x - 1] / a))
            # inductance compensation
            self.Db["hy_x"][r_x, r_y, :r_z+1] = (self.dt) / (u0_c1)
            self.Db["hx_y"][r_x, r_y, :r_z+1] = (self.dt) / (u0_c1)
            self.Db["hx_y"][r_x, r_y-1, :r_z+1] = (self.dt) / (u0_c1)
        # port faces -x
        elif self.pattern[r_x - 1, r_y] and not self.pattern[r_x , r_y]:
            print("-x")
            self.Db["hy_x"][r_x, r_y, r_z] = (2 * self.dt) / (u0_c1 * np.log(self.dx[r_x] / a))
            # inductance compensation
            self.Db["hy_x"][r_x-1, r_y, :r_z+1] = (self.dt) / (u0_c1)
            self.Db["hx_y"][r_x, r_y, :r_z+1] = (self.dt) / (u0_c1)
            self.Db["hx_y"][r_x, r_y-1, :r_z+1] = (self.dt) / (u0_c1)
        # shorten load along y direction
        # port faces +y
        elif not self.pattern[r_x, r_y - 1] and self.pattern[r_x, r_y]:
            print("+y")
            self.Db["hx_y"][r_x, r_y - 1, :r_z+1] = (2 * self.dt) / (u0_c1 * np.log(self.dy[r_y - 1] / a))
            # inductance compensation
            self.Db["hx_y"][r_x, r_y, :r_z+1] = (self.dt) / (u0_c1)
            self.Db["hy_x"][r_x-1, r_y, :r_z+1] = (self.dt) / (u0_c1)
            self.Db["hy_x"][r_x, r_y, :r_z+1] = (self.dt) / (u0_c1)
        # port faces -y
        elif self.pattern[r_x, r_y - 1] and not self.pattern[r_x , r_y]:
            print("-y")
            self.Db["hx_y"][r_x, r_y, :r_z+1] = (2 * self.dt) / (u0_c1 * np.log(self.dy[r_y]/ a))
            # inductance compensation
            self.Db["hx_y"][r_x, r_y - 1, :r_z+1] = (self.dt) / (u0_c1)
            self.Db["hy_x"][r_x-1, r_y, :r_z+1] = (self.dt) / (u0_c1)
            self.Db["hy_x"][r_x, r_y, :r_z+1] = (self.dt) / (u0_c1)
        
        # voltage sources
        self.ports[name] = dict(x=r_x, y=r_y, z=r_z, Vs_a = 1 / (denom * rterm))

        
    def generate_gif(self, filename, view="xz", vmax=30, vmin=-20, zoom=1.3, el=10, az=0, volume=False):
        nframe = len(self.gif_fields)
        # numpy type for the field values
        dtype_ = np.float32
        dx, dy, dz = self.dx, self.dy, self.dz
        Nx, Ny, Nz = len(dx), len(dy), len(dz)

        field = self.gif_fields[..., self.ms_z - 1]
        field = np.where(np.abs(field) < 1e-12, 1e-12, field)

        gx = np.concatenate([[0], np.cumsum(dx)])
        gy = np.concatenate([[0], np.cumsum(dy)])
        gz = np.concatenate([[0], np.cumsum(dz)])
        
        grid = pv.RectilinearGrid(gx, gy, (dz[0]))

        ez_fields = pv.RectilinearGrid(gx, gy, (dz[0]))

        plotter = pv.Plotter(off_screen=True)
        # add grid
        plotter.add_mesh(grid, style="wireframe", line_width=0.05, color="k", opacity=0.05)
        # add copper features
        pattern_g = pv.RectilinearGrid(gx, gy, dz[0])
        pattern_g.cell_data["values"] = self.pattern.flatten(order="F")

        # Create the colormap from the list of colors
        cmap_two_colors = mcolors.LinearSegmentedColormap.from_list(
            "custom_cmap", ["none", "gold"]
        )

        plotter.add_mesh(
            pattern_g, 
            opacity=1, 
            cmap=cmap_two_colors, 
            show_scalar_bar=False, 
            # smooth_shading=False, 
            # interpolate_before_map=False
        )
        plotter.add_axes()
        
        # substrate
        Gx, Gy, Gz = np.sum(dx), np.sum(dy), dz[0]
        
        sub = pv.Cube(
            center=(Gx / 2, Gy / 2, -Gz/2), 
            x_length=Gx, 
            y_length=Gy, 
            z_length=Gz
        )
        plotter.add_mesh(sub, opacity=0.1, color="green")

        data = 20 * np.log10(np.abs(field[20]))

        ez_fields.point_data['values'] = data.flatten()
        if volume:
            plotter.add_volume(ez_fields, cmap="jet", opacity="linear", scalars="values", clim=[vmin, vmax], show_scalar_bar=False)
        else:
            plotter.add_mesh(ez_fields, cmap="jet", opacity="linear", scalars="values", clim=[vmin, vmax], show_scalar_bar=False)

        plotter.render()    
        plotter.add_axes()
        plotter.camera_position = view
        # plotter.camera.position = (dx[0] * 120, -dy[0] * 80, dz[0] * 30)
        # plotter.camera.focal_point = (dx[0] * 120, dy[0] * 80, 0)
        plotter.camera.elevation += el
        plotter.camera.azimuth += az
        bar = plotter.add_scalar_bar(
            title="Ez [dB]\n", vertical=False, label_font_size=11, title_font_size=14
        )
        plotter.camera.zoom(zoom)

        plotter.open_gif(filename)

        for n in range(nframe):
            data = 20 * np.log10(np.abs(field[n]))
            data = np.clip(data, vmin, vmax)
            ez_fields.point_data["values"][:] = data.flatten(order="F")
            plotter.add_title(f"t={n * self.gif_step * self.dt * 1e9:.2f}ns")
            plotter.render()  
            plotter.write_frame()
        # return plotter

        # Closes and finalizes movie
        plotter.close()

    def get_ba(self, port, fields):

        x, y, z, _ = self.ports[port].values()

        dx, dy, dz = self.dx, self.dy, self.dz

        # half cell lengths between h components
        dx_h = (dx[1:] + dx[:-1]) / 2
        dy_h = (dy[1:] + dy[:-1]) / 2

        ez, hx, hy = fields[port]["ez"], fields[port]["hx"], fields[port]["hy"]

        # voltage in lumped port
        v1 = ez 

        # current in lumped port, defined as leaving the port
        c1 = (hy[:, 1] - hy[:, 0] ) * dy_h[y - 1]
        c2 = (-hx[:, 1] + hx[:, 0] ) * dx_h[x - 1]

        i1 = (c1 + c2) 

        # fig, ax = plt.subplots()
        # ax.plot(v1)

        # plt.figure()
        # plt.plot(v1)
        # plt.plot(i1 * 50)

        V1 = utils.dtft_f(v1[:800], self.frequency, 1 / self.dt)
        I1 = utils.dtft_f(i1[:800], self.frequency, 1 / self.dt)

        # delay current by half a time-step to be at the same time sample as the voltage
        # h components are ahead of the e components by half a time step
        I1 = I1 * np.exp(1j * 2 * np.pi * self.frequency * self.dt / 2)

        z0 = 50 
        A1 = (V1 + z0 * I1) / (2 * np.sqrt(z0.real))
        B1 = (V1 - np.conj(z0) * I1) / (2 * np.sqrt(z0.real))

        return B1, A1
    

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

dx0 = conv.m_in(0.02)
dy0 = conv.m_in(0.02)
dz0 = conv.m_in(0.01)

f0 = 10e9

Nx = 80
Ny = 35
Nz = 20

dx = np.ones(Nx) * dx0
dy = np.ones(Ny) * dy0
dz = np.ones(Nz) * dz0

eps_z = np.ones(Nz) * e0
eps_z[:2] = 3.66 * e0

# dz[:2] = conv.m_in(0.01)

Nx = 60 #int(( conv.m_in(2)/ dx0) + 20)
Ny = 60

dx = np.ones(Nx) * dx0
dy = np.ones(Ny) * dy0

p1_x, p1_y = 10, Ny//2

p2_x, p2_y = int((3/4) * Nx), Ny - 10
p3_x, p3_y = int((3/4) * Nx), 10

# p3_x, p3_y = 10, 23
# p4_x, p4_y = Nx - 10, 23

pattern = np.zeros((Nx, Ny), dtype=np.int32)
pattern[p1_x:p2_x, p1_y-1:p1_y+1] = 1
pattern[p2_x-1:p2_x+1, p1_y:] = 1
pattern[p3_x-1:p3_x+1, :p1_y+1] = 1
# pattern[p3_x, p1_y-2:p1_y+2] = 0

dz[4:] = conv.m_in(0.02)
dy[p1_y-3: p1_y+3] = conv.m_in(0.020)
dx[p2_x-2: p2_x+2] = conv.m_in(0.005)
# dy[p1_y+1] = conv.m_in(0.005)

fig, ax = plt.subplots()
ax.pcolormesh(pattern)


s = Solver_SingleLayer(frequency, pattern, dx, dy, dz, eps_z, ms_z=2)

s.add_port("p1", p1_x, p1_y)
s.add_port("p2", p2_x, p2_y)
s.add_port("p3", p3_x, p3_y)
s.add_xPML(d_pml = 10, side="upper")
s.add_yPML(d_pml = 10, side="lower")
s.add_yPML(d_pml = 10, side="upper")
s.add_zPML(d_pml = 5)

# a = conv.m_in(1e-12)
# s.Db["hz_y"][p1_x:, p1_y-2, 1] = (2 * s.dt) / (u0 * np.log(s.dy[p1_y-2] / a))
# s.Db["hz_y"][p1_x:, p1_y+1, 1] = (2 * s.dt) / (u0 * np.log(s.dy[p1_y+1] / a))



s.add_c_probe("probe1", 25, p1_y-2, p1_y+1)
s.add_v_probe("probe1", 25, p1_y)


Nt = 1300
f0 = 10e9
src = np.zeros(Nt)
pulse_n = 1000
# width of half pulse in time
t_half = 4e-11#(dt * (pulse_n // 8))
# center of the pulse in time
t0 = (s.dt * 340)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
# gaussian modulated sine wave source
a = 2e-2
src[:pulse_n] = a * (np.sin(2*np.pi*f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32).squeeze()
# src[:pulse_n] = a * (np.exp(-((t - t0) / t_half)**2)).astype(dtype_).squeeze()
plt.figure()
plt.plot(src)

fields = s.run("p1", src, gif_step=15)

# # probe fields


vp = s.v_probes["probe1"]["values"]
cp = s.c_probes["probe1"]["values"]

# plt.plot(vp)
# plt.plot(cp *50)

Vp = utils.dtft_f(vp, s.frequency, 1 / s.dt)
Ip = utils.dtft_f(cp, s.frequency, 1 / s.dt)

Ip = Ip * np.exp(1j * 2 * np.pi * s.frequency * s.dt / 2)

Zp = Vp / Ip

# fig, ax = plt.subplots()
# plt.plot(Zp)

# fwd /= np.max(np.abs(fwd))

# ax.plot(np.arange(len(src)) * s.dt / 1e-9, conv.db20_lin(fwd))
# ax.set_ylim([-80, 0])
# ax.set_xlabel("Time [ns]")
# ax.set_ylabel("|$E_z$| [dB]")
# ax.grid(True)



# sparameters
b1, a1 = s.get_ba("p1", fields)
# b2, a2 = s.get_ba("p2", fields)

# fig, ax = plt.subplots()
# ax_t = ax.twinx()
# ax.plot(frequency  / 1e9, conv.db20_lin(b1 / a1))
# # ax_t.plot(frequency / 1e9, conv.db20_lin(b2 / a1), color="orange", alpha=0.5)
# ax.set_xlabel("Frequency [GHz]")
# ax.set_ylim([-50, 1])
# ax.margins(x=0)
# ax.legend(["S11", "S21"], loc="upper left")
# ax_t.legend(["S21"], loc="upper right")
# ax.set_ylabel("dB")
# ax_t.set_ylabel("dB")
# mplm.axis_marker(y=np.max(conv.db20_lin(b1 / a1)), axes=ax)
# plt.show()

S11 = b1 / a1
fig, ax = plt.subplots()
rfn.plots.draw_smithchart(ax)
plt.plot(S11.real, S11.imag)

fig, ax = plt.subplots()
ax.plot(frequency / 1e9, conv.db20_lin(S11))
ax.set_ylim([-50, 0])
ax.set_xlabel("Frequency [GHz]")

# fig, ax = plt.subplots()
# plt.plot(frequency / 1e9, conv.db20_lin(S11))

msline50 = rfn.elements.MSLine(
    w=0.011, 
    h=0.020, 
    er=3.66, 
)

zref = msline50.get_properties(10e9).sel(value="z0")[0]

z.real
# plot impedance
z = conv.z_gamma(S11)
fig, ax = plt.subplots()
ax.plot(frequency/1e9, z.real, alpha=0.3)
ax.plot(frequency/1e9, Zp)
ax.set_ylim([0, 120])
mplm.line_marker(x=10)
mplm.axis_marker(y=zref)
ax.set_xlabel("Frequency [GHz]")



pv.set_jupyter_backend('trame')
p = s.generate_gif("msline_2.gif", el=0, zoom=1.5, vmin=0, vmax=30, az=0, view="xy", volume=False)
# p.show()
ipyimage(filename='msline_2.gif')


# plt.pcolormesh(s.Cb["ey_x"][:, :, 1].T)

# s.Cb["ex_y"][25, :, 1]