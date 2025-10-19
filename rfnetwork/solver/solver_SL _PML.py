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


class Solver_SingleLayer():

    def __init__(
        self, 
        frequency: np.ndarray,
        pattern: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        er: float, 
        sub_h: float,
        Nz: int = None
    ):
        self.frequency = frequency
        self.dx = dx
        self.dy = dy

        # maximum frequency in the source
        fmax = np.max(frequency)

        # smallest wavelength in the substrate material
        vp = c0 / np.sqrt(er)
        lam_sub = vp / fmax

        # largest allowed size of cell in substrate, require at least 10 cells per wavelength
        cells_per_wavelength = 10
        sub_dmax = (lam_sub / cells_per_wavelength)

        # number of cells in the z direction inside the substrate, typically 1
        sub_Nz = int((sub_h // sub_dmax) + 1)

        if sub_Nz > 1:
            raise NotImplementedError("Fix ports for multi-cell substrates")
        
        # z-index of microstrip features 
        ms_z = sub_Nz
        # z-index of substrate layer
        sub_z = slice(0, sub_Nz)

        # number of total cells in the x, y, z direction
        Nx = pattern.shape[0]
        Ny = pattern.shape[1]
        
        if Nz is None:
            Nz = int(sub_Nz * 15)

        # cell sizes in z direction
        self.dz = np.ones(Nz) * sub_h

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
        sub_eps = er * e0
        Cb_ez[..., sub_z] = (2 * dt) / ((2 * sub_eps + (sig_0 * dt))) 
        Cb_ex[..., sub_z] = (2 * dt) / ((2 * sub_eps + (sig_0 * dt))) 
        Cb_ey[..., sub_z] = (2 * dt) / ((2 * sub_eps + (sig_0 * dt))) 

        # ex and ey are on the boundary between material cells, compute the average of the substrate
        # FIX for non-uniform grids
        Cb_ex[..., ms_z] = (Cb_ex[..., ms_z - 1] + Cb_ex[..., ms_z + 1]) / 2
        Cb_ey[..., ms_z] = (Cb_ey[..., ms_z - 1] + Cb_ey[..., ms_z + 1]) / 2

        # PEC pattern
        # conductivity of PEC
        sig_pec = 1e7
        # effective permittivity for ex/ey components on the boundary of the substrate
        er_eff = (sub_eps + e0) / 2

        # set ex coefficient to PEC if either cell next to it (along y) is set as copper
        ex_patt = (pattern[:, :-1] | pattern[:, 1:]) 
        ex_sig = ex_patt * sig_pec

        Ca_ex[..., 1:-1, ms_z] = np.where(
            ex_patt, (2 * er_eff - (ex_sig * dt)) / (2 * er_eff + (ex_sig * dt)), Ca_ex[..., 1:-1, ms_z]
        )
        Cb_ex[..., 1:-1, ms_z] = np.where(
            ex_patt, (2 * dt) / ((2 * er_eff + (ex_sig * dt))), Cb_ex[..., 1:-1, ms_z]
        )

        # set ey coefficient to PEC if either cell next to it (along x) is set as copper
        ey_patt = (pattern[:-1] | pattern[1:]) 
        ey_sig = ey_patt * sig_pec
        
        Ca_ey[1:-1, :, ms_z] = np.where(
            ey_patt, (2 * er_eff - (ey_sig * dt)) / (2 * er_eff + (ey_sig * dt)), Ca_ey[1:-1, :, ms_z]
        )
        Cb_ey[1:-1, :, ms_z] = np.where(
            ey_patt, (2 * dt) / ((2 * er_eff + (ey_sig * dt))) , Cb_ey[1:-1, :, ms_z]
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
        self.er = er
        self.dt = dt
        self.pattern = pattern
        self.ports = dict()

    def run(self, port, v_waveform, gif_step=10):

        Nt = len(v_waveform)

        # numpy type for the field values
        dtype_ = np.float32
        dx, dy, dz = self.dx, self.dy, self.dz
        Nx, Ny, Nz = len(dx), len(dy), len(dz)
        
        self.gif_step = gif_step
        self.gif_fields = np.zeros((Nt // self.gif_step, Nx+1, Ny+1, Nz), dtype=dtype_)

        
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
                port_fields[k]["ez"][n + 1] = ez[x, y, z]
                port_fields[k]["hx"][n + 1] = hx[x, y-1: y+1, z]
                port_fields[k]["hy"][n + 1] = hy[x-1: x+1, y, z]

            if (n % self.gif_step) == 0:
                self.gif_fields[n // self.gif_step] = ez

        return port_fields


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
        sigma_e_n = sigma_max * ((i_pml) / (d_pml + 0.5))**m_pml
        # sigma in the middle of the cells. First Hz component in the PML is 0.5 cells into the PML
        sigma_e_np5 = sigma_max * ((i_pml + 0.5) / (d_pml + 0.5))**m_pml

        # magnetic conductivity
        sigma_m_n = sigma_e_n * u0 / e0
        sigma_m_np5 = sigma_e_np5 * u0 / e0

        plt.figure()
        plt.plot(np.arange(0, 10, 1), sigma_m_n.squeeze())
        plt.plot(np.arange(0.5, 10.5, 1), sigma_m_np5.squeeze())

        # first ex component is at the edge of the PML where sigma = 0, last component is at the solve boundary and not updated
        self.Ca["ex_z"][..., -d_pml-1:-1] = (2 * e0 - (sigma_e_n * dt)) / (2 * e0 + (sigma_e_n * dt))
        self.Cb["ex_z"][..., -d_pml-1:-1] = (2 * dt) / ((2 * e0 + (sigma_e_n * dt))) 

        self.Ca["ey_z"][..., -d_pml-1:-1] = (2 * e0 - (sigma_e_n * dt)) / (2 * e0 + (sigma_e_n * dt))
        self.Cb["ey_z"][..., -d_pml-1:-1] = (2 * dt) / ((2 * e0 + (sigma_e_n * dt))) 

        # hx/hy components are in the middle of the PML cells, use half cell indices
        self.Da["hx_z"][..., -d_pml:] = (2 * u0 - (sigma_m_np5 * dt)) / (2 * u0 + (sigma_m_np5 * dt))
        self.Db["hx_z"][..., -d_pml:] = (2 * dt) / ((2 * u0 + (sigma_m_np5 * dt))) 
        
        self.Da["hy_z"][..., -d_pml:] = (2 * u0 - (sigma_m_np5 * dt)) / (2 * u0 + (sigma_m_np5 * dt))
        self.Db["hy_z"][..., -d_pml:] = (2 * dt) / ((2 * u0 + (sigma_m_np5 * dt))) 

        # plt.plot(self.Da["hy_z"][10, 10])
        # plt.plot(self.Db["hy_z"][10, 10])
        
        # Ca_0 = (2 * e0 - (sig_0 * dt)) / (2 * e0 + (sig_0 * dt))
        # Cb_0 = (2 * dt) / ((2 * e0 + (sig_0 * dt))) 
        # Da = (2 * u0 - (sigma_m * dt)) / (2 * u0 + (sigma_m * dt))
        # Db_0 = (2 * dt) / ((2 * u0 + (sigma_m * dt))) 
        

    def add_port(self, name, ez_x, ez_y, r0=50, ind_comp = 0.75):
        # resistive load
        r0 = 50
        r_x = ez_x

        r_y = ez_y
        r_z = self.ms_z - 1

        dx_r = (self.dx[ez_x - 1] + self.dx[ez_x]) / 2
        dy_r = (self.dy[ez_y - 1] + self.dy[ez_y]) / 2
        dz_r = self.dz[r_z]

        rterm = (r0 * dx_r * dy_r)

        denom = (self.er * e0 / self.dt) + (dz_r / (2 * rterm))

        # ez component for resistor
        self.Ca["ez_x"][r_x, r_y, r_z] = ((self.er * e0 / self.dt) - (dz_r / (2 * rterm))) / denom
        self.Cb["ez_x"][r_x, r_y, r_z] = 1 / (denom)

        self.Ca["ez_y"][r_x, r_y, r_z] = ((self.er * e0 / self.dt) - (dz_r / (2 * rterm))) / denom
        self.Cb["ez_y"][r_x, r_y, r_z] = 1 / (denom)

        # TODO: add wires for multi-cell 
        # 0 ohm resistors to connect the lumped element to the trace
        # Ca_ez[r_x-1, r_y-1, 0] = -1
        # Cb_ez[r_x-1, r_y-1, 0] = 0

        # Ca_ez[r_x-1, r_y-1, 2] = -1
        # Cb_ez[r_x-1, r_y-1, 2] = 0
        ###

        # h components around load, thin wire model
        a = conv.m_in(1e-9)
        u0_c1 = u0 * ind_comp

        # shorten load along the x direction
        # port faces +x
        if not self.pattern[r_x - 1, r_y] and self.pattern[r_x, r_y]:
            print("+x")
            self.Db["hy_x"][r_x - 1, r_y, r_z] = (2 * self.dt) / (u0_c1 * np.log(self.dx[r_x - 1] / a))
            # inductance compensation
            self.Db["hy_x"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hx_y"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hx_y"][r_x, r_y-1, r_z] = (self.dt) / (u0_c1)
        # port faces -x
        elif self.pattern[r_x - 1, r_y] and not self.pattern[r_x , r_y]:
            print("-x")
            self.Db["hy_x"][r_x, r_y, r_z] = (2 * self.dt) / (u0_c1 * np.log(self.dx[r_x] / a))
            # inductance compensation
            self.Db["hy_x"][r_x-1, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hx_y"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hx_y"][r_x, r_y-1, r_z] = (self.dt) / (u0_c1)
        # shorten load along y direction
        # port faces +y
        elif not self.pattern[r_x, r_y - 1] and self.pattern[r_x, r_y]:
            print("+y")
            self.Db["hx_y"][r_x, r_y - 1, r_z] = (2 * self.dt) / (u0_c1 * np.log(self.dy[r_y - 1] / a))
            # inductance compensation
            self.Db["hx_y"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hy_x"][r_x-1, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hy_x"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
        # port faces -y
        elif self.pattern[r_x, r_y - 1] and not self.pattern[r_x , r_y]:
            print("-y")
            self.Db["hx_y"][r_x, r_y, r_z] = (2 * self.dt) / (u0_c1 * np.log(self.dy[r_y]/ a))
            # inductance compensation
            self.Db["hx_y"][r_x, r_y - 1, r_z] = (self.dt) / (u0_c1)
            self.Db["hy_x"][r_x-1, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hy_x"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
        
        # voltage sources
        self.ports[name] = dict(x=r_x, y=r_y, z=r_z, Vs_a = 1 / (denom * rterm))

        

    

    def generate_gif(self, filename):
        nframe = len(self.gif_fields)
        # numpy type for the field values
        dtype_ = np.float32
        dx, dy, dz = self.dx, self.dy, self.dz
        Nx, Ny, Nz = len(dx), len(dy), len(dz)

        field = self.gif_fields[:, :Nx, :Ny, :Nz]
        g = pv.ImageData()

        grid =  np.ones((Nx, Ny, Nz))
        g.dimensions = (Nx, Ny, Nz)
        dmax = dx[0]
        g.spacing = (dx[0], dy[0], dz[0])
        # edges = g.extract_all_edges()

        # Open a gif
        plotter = pv.Plotter(off_screen=True)

        plotter.add_mesh(g, style="wireframe", line_width=0.05, color="k", opacity=0.05)
        # plotter.show()

        pattern_g = pv.ImageData(dimensions=self.pattern.shape + (2,))
        pattern_g.spacing = (dx[0], dy[0], dz[0])

        # patt_nan = np.where(self.pattern < 0.01, np.nan, self.pattern)[..., None]
        p = np.broadcast_to(self.pattern[..., None], self.pattern.shape + (2,)).copy()
        p[..., 0] = 0
        pattern_g.point_data["scalars"] = p.flatten(order="F")
        # trace_pnts = np.array([(ms_x.start, ms_y.start, ms_z), (ms_x.stop, ms_y.start, ms_z), (ms_x.stop, ms_y.stop, ms_z)])
        # trace = pv.Rectangle(trace_pnts * dmax)

        # Create the colormap from the list of colors
        cmap_two_colors = mcolors.LinearSegmentedColormap.from_list(
            "custom_cmap", ["none", "gold"]
        )

        plotter.add_mesh(pattern_g, opacity=1, cmap=cmap_two_colors, show_scalar_bar=False, smooth_shading=False, interpolate_before_map=False)

        gx0, dy0, dz0 = (dx[0], dy[0], dz[0])
        sub = pv.Cube(np.array(((Nx//2 - 0.5) * dx0, (Ny//2 - 0.5) * dy0, 0.5 * dz0)), (Nx-1) * dx0, (Ny-1) * dy0, dz0)
        plotter.add_mesh(sub, opacity=0.1, color="green")


        data = 20 * np.log10(np.abs(field[20]))

        vmin = -20
        vmax = 30
        data = np.clip(data, vmin, vmax)

        g.point_data['values'] = data.flatten(order="F")
        plotter.add_volume(
            g, cmap="jet", opacity="linear", scalars="values", clim=[vmin, vmax], show_scalar_bar=False
        )

        # data = 20 * np.log10(np.abs(ez[40]))
        # data = np.clip(data, -80, -20)

        # g.point_data["values"][:] = data.flatten(order="F")     # update in-place                 # mark as modified
        plotter.camera.zoom(1)
        plotter.render()    
        plotter.add_axes()
        # plotter.add_bounding_box()
        plotter.camera_position = "xz"
        plotter.camera.elevation += 10
        # plotter.camera.azimuth += 3
        plotter.camera.zoom(1.3)
        bar = plotter.add_scalar_bar(
            title="Ez [dB]\n", vertical=False, label_font_size=11, title_font_size=14
        )
        # bar.GetTitleTextProperty().SetLineSpacing(3)
        # plotter.show()


        plotter.open_gif(filename)

        for n in range(nframe):
            data = 20 * np.log10(np.abs(field[n]))
            data = np.clip(data, vmin, vmax)
            g.point_data["values"][:] = data.flatten(order="F")
            plotter.add_title(f"t={n * self.gif_step * self.dt * 1e9:.2f}ns")
            plotter.render()  
            plotter.write_frame()
        # # plotter.show()

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
        v1 = -ez * self.dz[z]

        # current in lumped port, defined as leaving the port
        c1 = (hy[:, 1] - hy[:, 0] ) * dy_h[y - 1]
        c2 = (-hx[:, 1] + hx[:, 0] ) * dx_h[x - 1]

        i1 = (c1 + c2)

        fig, ax = plt.subplots()
        ax.plot(v1)

        # plt.figure()
        # plt.plot(v1)
        # plt.plot(i1 * 50)

        V1 = utils.dtft_f(v1, self.frequency, 1 / self.dt)
        I1 = utils.dtft_f(i1, self.frequency, 1 / self.dt)

        # advance current by half a time-step to be at the same time sample as the voltage
        # h components are behind of the e components by half a time step
        I1 = I1 * np.exp(1j * 2 * np.pi * self.frequency * self.dt / 2)

        z0 = 50 
        A1 = (V1 + z0 * I1) / (2 * np.sqrt(z0.real))
        B1 = (V1 - np.conj(z0) * I1) / (2 * np.sqrt(z0.real))

        return B1, A1
    

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)
er: float = 3.66
er_eff = 2.84
sub_h = conv.m_in(0.02)

dx0 = conv.m_in(0.02)
dy0 = conv.m_in(0.02)


f0 = 10e9
lam0 = (const.c0 / np.sqrt(er_eff)) / f0

Nx = int(( conv.m_in(2)/ dx0) + 20)
Ny = 20

dx = np.ones(Nx) * dx0
dy = np.ones(Ny) * dy0

p1_x, p1_y = 10, Ny//2
p2_x, p2_y = Nx - 10, Ny//2

# p3_x, p3_y = 10, 23
# p4_x, p4_y = Nx - 10, 23

pattern = np.zeros((Nx, Ny), dtype=np.int32)
pattern[p1_x: p2_x, p1_y-1:p1_y+1] = 1
# pattern[p3_x: p4_x, p3_y-1:p3_y+1] = 1
# pattern[p1_x:, p1_y-1:p1_y+1] = 1
    
dx = np.ones(Nx) * dx0
dy = np.ones(Ny) * dy0


s = Solver_SingleLayer(frequency, pattern, dx, dy, er, sub_h, 20)

s.add_port("p1", p1_x, p1_y)
s.add_port("p2", p2_x, p2_y)
# s.add_zPML()


Nt = 1300
f0 = 10e9
src = np.zeros(Nt)
pulse_n = 1000
# width of half pulse in time
t_half = 9e-11#(dt * (pulse_n // 8))
# center of the pulse in time
t0 = (s.dt * 250)

t = np.linspace(0, s.dt * pulse_n, pulse_n)
# gaussian modulated sine wave source
a = 1e-2
src[:pulse_n] = a * (np.sin(2*np.pi*f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32).squeeze()
# src[:pulse_n] = a * (np.exp(-((t - t0) / t_half)**2)).astype(dtype_).squeeze()
plt.figure()
plt.plot(src)

fields = s.run("p1", src)

b1, a1 = s.get_ba("p1", fields)
b2, a2 = s.get_ba("p2", fields)

fig, ax = plt.subplots()
ax_t = ax.twinx()
ax.plot(frequency  / 1e9, conv.db20_lin(b1 / a1))
ax_t.plot(frequency / 1e9, conv.db20_lin(b2 / a1), color="orange", alpha=0.5)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylim([-50, 1])
ax.margins(x=0)
ax.legend(["S11", "S21"], loc="upper left")
ax_t.legend(["S21"], loc="upper right")
ax.set_ylabel("dB")
ax_t.set_ylabel("dB")
mplm.line_marker(x=10)
plt.show()



import time
stime = time.time()
fields = s.run("p1", src)
print(time.time() - stime)

s.generate_gif("msline_2.gif")
ipyimage(filename='msline_2.gif')
