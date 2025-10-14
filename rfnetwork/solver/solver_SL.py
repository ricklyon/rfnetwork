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
        Nz = int(sub_Nz * 20)

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
        ex_sig = (pattern[:, :-1] | pattern[:, 1:]) * sig_pec
        Ca_ex[..., 1:-1, ms_z] = (2 * er_eff - (ex_sig * dt)) / (2 * er_eff + (ex_sig * dt))
        Cb_ex[..., 1:-1, ms_z] = (2 * dt) / ((2 * er_eff + (ex_sig * dt))) 

        # set ey coefficient to PEC if either cell next to it (along x) is set as copper
        ey_sig = (pattern[:-1] | pattern[1:]) * sig_pec
        Ca_ey[1:-1, :, ms_z] = (2 * er_eff - (ey_sig * dt)) / (2 * er_eff + (ey_sig * dt))
        Cb_ey[1:-1, :, ms_z] = (2 * dt) / ((2 * er_eff + (ey_sig * dt))) 

        self.Ca = dict(
            ex = Ca_ex,
            ey = Ca_ey,
            ez = Ca_ez,
        )
    
        self.Cb = dict(
            ex = Cb_ex,
            ey = Cb_ey,
            ez = Cb_ez,
        )

        self.Db = dict(
            hx_ey = np.ones((Nx+1, Ny, Nz)) * Db_0,
            hx_ez = np.ones((Nx+1, Ny, Nz)) * Db_0,
            hy_ez = np.ones((Nx, Ny+1, Nz)) * Db_0,
            hy_ex = np.ones((Nx, Ny+1, Nz)) * Db_0,
            hz_ex = np.ones((Nx, Ny, Nz+1)) * Db_0,
            hz_ey = np.ones((Nx, Ny, Nz+1)) * Db_0,
        )

        self.ms_z = ms_z
        self.er = er
        self.dt = dt
        self.pattern = pattern
        self.ports = dict()

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
        self.Ca["ez"][r_x, r_y, r_z] = ((self.er * e0 / self.dt) - (dz_r / (2 * rterm))) / denom
        self.Cb["ez"][r_x, r_y, r_z] = 1 / (denom)

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
            self.Db["hy_ez"][r_x - 1, r_y, r_z] = (2 * self.dt) / (u0_c1 * np.log(self.dx[r_x - 1] / a))
            # inductance compensation
            self.Db["hy_ez"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hx_ez"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hx_ez"][r_x, r_y-1, r_z] = (self.dt) / (u0_c1)
        # port faces -x
        elif self.pattern[r_x - 1, r_y] and not self.pattern[r_x , r_y]:
            print("-x")
            self.Db["hy_ez"][r_x, r_y, r_z] = (2 * self.dt) / (u0_c1 * np.log(self.dx[r_x] / a))
            # inductance compensation
            self.Db["hy_ez"][r_x-1, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hx_ez"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hx_ez"][r_x, r_y-1, r_z] = (self.dt) / (u0_c1)
        # shorten load along y direction
        # port faces +y
        elif not self.pattern[r_x, r_y - 1] and self.pattern[r_x, r_y]:
            print("+y")
            self.Db["hx_ez"][r_x, r_y - 1, r_z] = (2 * self.dt) / (u0_c1 * np.log(self.dy[r_y - 1] / a))
            # inductance compensation
            self.Db["hx_ez"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hy_ez"][r_x-1, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hy_ez"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
        # port faces -y
        elif self.pattern[r_x, r_y - 1] and not self.pattern[r_x , r_y]:
            print("-y")
            self.Db["hx_ez"][r_x, r_y, r_z] = (2 * self.dt) / (u0_c1 * np.log(self.dy[r_y]/ a))
            # inductance compensation
            self.Db["hx_ez"][r_x, r_y - 1, r_z] = (self.dt) / (u0_c1)
            self.Db["hy_ez"][r_x-1, r_y, r_z] = (self.dt) / (u0_c1)
            self.Db["hy_ez"][r_x, r_y, r_z] = (self.dt) / (u0_c1)
        
        # voltage sources
        self.ports[name] = dict(x=r_x, y=r_y, z=r_z, Vs_a = 1 / (denom * rterm))


    def run(self, port, v_waveform):

        Nt = len(v_waveform)
        # numpy type for the field values
        dtype_ = np.float32
        dx, dy, dz = self.dx, self.dy, self.dz
        Nx, Ny, Nz = len(dx), len(dy), len(dz)

        # field values
        ex = np.zeros((Nt, Nx, Ny+1, Nz+1), dtype=dtype_)
        ey = np.zeros((Nt, Nx+1, Ny, Nz+1), dtype=dtype_)
        ez = np.zeros((Nt, Nx+1, Ny+1, Nz), dtype=dtype_)
        hx = np.zeros((Nt, Nx+1, Ny, Nz), dtype=dtype_)
        hy = np.zeros((Nt, Nx, Ny+1, Nz), dtype=dtype_)
        hz = np.zeros((Nt, Nx, Ny, Nz+1), dtype=dtype_)

        Ca_ex = self.Ca["ex"][:, 1:-1, 1:-1], # edges along y and z do not get updated
        Ca_ey = self.Ca["ey"][1:-1, :, 1:-1], # edges along x and z do not get updated
        Ca_ez = self.Ca["ez"][1:-1, 1:-1, :]  # edges along x and y do not get updated

        Cb_ex = self.Cb["ex"][:, 1:-1, 1:-1], # edges along y and z do not get updated
        Cb_ey = self.Cb["ey"][1:-1, :, 1:-1], # edges along x and z do not get updated
        Cb_ez = self.Cb["ez"][1:-1, 1:-1, :]  # edges along x and y do not get updated

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

        src_x, src_y, src_z, Vs_a = self.ports[port].values()

        stime = time.time()
        # loop over each time step
        for n in range(Nt -  1):

            # grid starts at bottom left corner. A half-cell is placed in front of each component so all field values have 
            # the same number of values. The extra half cell components are not updated.

            # ex update
            # edges along y and z do not get updated
            ex[n+1, :, 1:-1, 1:-1] = (Ca_ex * ex[n, :, 1:-1, 1:-1]) + Cb_ex * (
                (np.diff(hz[n], axis=1) * dy_h_inv)[:, :, 1:-1] - 
                (np.diff(hy[n], axis=2) * dz_h_inv)[:, 1:-1, :]
            )

            # ey update
            # edges along x and z do not get updated
            ey[n+1, 1:-1, :, 1:-1] = (Ca_ey * ey[n, 1:-1, :, 1:-1]) + Cb_ey * (
                (np.diff(hx[n], axis=2) * dz_h_inv)[1:-1, :, :] - 
                (np.diff(hz[n], axis=0) * dx_h_inv)[:, :, 1:-1]
            )

            # ez update
            # edges along x and y do not get updated
            ez[n+1, 1:-1, 1:-1, :] = (Ca_ez * ez[n, 1:-1, 1:-1, :]) + Cb_ez * (
                (np.diff(hy[n], axis=0) * dx_h_inv)[:, 1:-1, :] - 
                (np.diff(hx[n], axis=1) * dy_h_inv)[1:-1, :, :]
            )

            # add resistive voltage source
            # Vs_a is already divided by sub_z, so we don't need to split the voltage among each ez component
            ez[n+1, src_x, src_y, src_z] -= Vs_a * (v_waveform[n + 1]) # book says src/2

            # hx update
            # all edges are updated
            hx[n+1, :, :, :] = (hx[n, :, :, :]) + (
                self.Db["hx_ey"] * (np.diff(ey[n+1], axis=2) * dz_inv) - 
                self.Db["hx_ez"] * (np.diff(ez[n+1], axis=1) * dy_inv)
            )

            # hy update
            # all edges are updated
            hy[n+1, :, :, :] = (hy[n, :, :, :]) + (
                self.Db["hy_ez"]  * (np.diff(ez[n+1], axis=0) * dx_inv) - 
                self.Db["hy_ex"]  * (np.diff(ex[n+1], axis=2) * dz_inv)
            )

            # hz update
            # all edges are updated
            hz[n+1, :, :, :] = (hz[n, :, :, :]) + (
                self.Db["hz_ex"]  * (np.diff(ex[n+1], axis=1) * dy_inv) - 
                self.Db["hz_ey"]  * (np.diff(ey[n+1], axis=0) * dx_inv)
            )

        return dict(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz)

    def generate_gif(self, field):
        Nt = len(field)
        # numpy type for the field values
        dtype_ = np.float32
        dx, dy, dz = self.dx, self.dy, self.dz
        Nx, Ny, Nz = len(dx), len(dy), len(dz)

        field = field[:, :Nx, :Ny, :Nz]
        g = pv.ImageData()

        grid =  np.ones((Nx, Ny, Nz))
        g.dimensions = (Nx, Ny, Nz)
        dmax = dx[0]
        g.spacing = (dmax, dmax, dmax)
        # edges = g.extract_all_edges()

        # Open a gif
        plotter = pv.Plotter(off_screen=True)

        plotter.add_mesh(g, style="wireframe", line_width=0.05, color="k", opacity=0.05)
        # plotter.show()

        pattern_g = pv.ImageData(dimensions=self.pattern.shape + (2,))
        pattern_g.spacing = (dmax, dmax, dmax)

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

        sub = pv.Cube(np.array((Nx//2 - 0.5, Ny//2 - 0.5, 0.5)) * dmax, (Nx-1) * dmax, (Ny-1) * dmax, dmax)
        plotter.add_mesh(sub, opacity=0.2, color="green")


        data = 20 * np.log10(np.abs(field[50]))

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
        plotter.camera.elevation += 30
        plotter.camera.azimuth += 10
        plotter.camera.zoom(1.3)
        bar = plotter.add_scalar_bar(
            title="Ez [dB]\n", vertical=False, label_font_size=11, title_font_size=14
        )
        # bar.GetTitleTextProperty().SetLineSpacing(3)
        # plotter.show()


        plotter.open_gif('outputs/msline_2.gif')
        nstep = 15
        nframe = Nt // nstep
        for n in range(nframe):
            data = 20 * np.log10(np.abs(field[n*nstep]))
            data = np.clip(data, vmin, vmax)
            g.point_data["values"][:] = data.flatten(order="F")
            plotter.add_title(f"t={n * nstep * self.dt * 1e9:.2f}ns")
            plotter.render()  
            plotter.write_frame()
        # # plotter.show()

        # Closes and finalizes movie
        plotter.close()


frequency: np.ndarray = np.arange(5e9, 15.01e9, 10e6)
er: float = 3.66
sub_h = conv.m_in(0.02)

dx0 = conv.m_in(0.02)
dy0 = conv.m_in(0.02)

# Nx = int((conv.m_in(2) / dx0) + 20)
# Ny = 20

# dx = np.ones(Nx) * dx0
# dy = np.ones(Ny) * dy0

# p1_x, p1_y = 10, (Ny // 2)
# p2_x, p2_y = Nx - 10, (Ny // 2)

# pattern = np.zeros((Nx, Ny), dtype=np.int32)
# pattern[p1_x: p2_x, p1_y-1:p1_y+1] = 1

Nx = 80
Ny = 80

dx = np.ones(Nx) * dx0
dy = np.ones(Ny) * dy0

p1_x, p1_y = 10, 20
p2_x, p2_y = Nx-20, Ny - 20

pattern = np.zeros((Nx, Ny), dtype=np.int32)
pattern[p1_x: p2_x, p1_y-1:p1_y+1] = 1
pattern[p2_x-1: p2_x+1, p1_y:p2_y] = 1



yy, xx = np.meshgrid(np.arange(Ny), np.arange(Nx))
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(xx, yy, pattern)
ax.grid()
# ax.set_yticks(np.arange(0.5, Ny + 0.5, 1))
ax.set_aspect("equal")

s = Solver_SingleLayer(frequency, pattern, dx, dy, er, sub_h)

s.add_port("p1", p1_x, p1_y)
s.add_port("p2", p2_x, p2_y)

s.dt

Nt = 1200
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

ez = s.run("p2", src)["ez"]

s.generate_gif(ez)
ipyimage(filename='outputs/msline_2.gif')