
import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time

from IPython.display import Image as ipyimage
import rfnetwork as rfn
import mpl_markers as mplm

u0 = const.u0
e0 = const.e0
c0 = const.c0

msline50 = rfn.elements.MSLine(
    w=0.040, 
    h=0.020, 
    er=3.57, 
)

msline50.get_properties(10e9)

# number of cells in each dimension
Nx = 108
Ny = 30
Nz = 20

Nt = 2500
f0 = 10e9
fmax = 20e9

spatial_shape = (Nx, Ny, Nz)

max_er: float = 3,
# minimum number of cells per wavelength
cells_per_wavelength: int = 15
dtype_ = np.float32

# smallest wavelength
vp = c0 / np.sqrt(max_er)
lam_min = vp / fmax

# largest allowed cell length
dmax = lam_min / cells_per_wavelength
conv.in_m(dmax)

# field values
ex = np.zeros((Nt, Nx, Ny+1, Nz+1), dtype=dtype_)
ey = np.zeros((Nt, Nx+1, Ny, Nz+1), dtype=dtype_)
ez = np.zeros((Nt, Nx+1, Ny+1, Nz), dtype=dtype_)
hx = np.zeros((Nt, Nx+1, Ny, Nz), dtype=dtype_)
hy = np.zeros((Nt, Nx, Ny+1, Nz), dtype=dtype_)
hz = np.zeros((Nt, Nx, Ny, Nz+1), dtype=dtype_)

# cell sizes
dx = np.ones(Nx) * dmax
dy = np.ones(Ny) * dmax
dz = np.ones(Nz) * dmax

# material properties
epsilon = np.ones(spatial_shape, dtype=dtype_) * e0
mu = np.ones(spatial_shape, dtype=dtype_) * u0
mu_m = np.zeros(spatial_shape, dtype=dtype_)
sigma = np.zeros(spatial_shape, dtype=dtype_)
sigma_m = np.zeros(spatial_shape, dtype=dtype_)

y_mid = Ny // 2

ms_y_mid = y_mid + 1
ms_y = slice(y_mid, y_mid + 3)
ms_y_ex = slice(y_mid, y_mid + 4)
ms_x = slice(10, -10)
ms_z = 3
sub_z = slice(0, ms_z)

w = 0.040
h = 0.02
sub_er = 3.57

# trace
dy[ms_y] = conv.m_in(w / 3)
# dz[5] = conv.m_in(cu_h)
# sigma[10:-10, ms_y, 5] = 6e7

# substrate
dz[sub_z] = conv.m_in(h / ms_z)
epsilon[:, :, sub_z] = sub_er * e0
(epsilon / e0)[0, 0]
# sigma[:, :, sub_z] = 0.04


# compute maximum time step that ensures convergence, use freespace propagation speed as worst case
length_min = np.array([np.min(dx), np.min(dy), np.min(dz)])
dmin = 1 / np.sqrt(((1 / length_min)**2).sum())
# S = 0.80 * (1 / np.sqrt(3))
dt = 0.9 * (dmin / const.c0)
conv.in_m(dx)

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

# source
src = np.zeros(Nt)
pulse_n = 1000
# width of half pulse in time
t_half = 7e-11#(dt * (pulse_n // 8))
# center of the pulse in time
t0 = (dt * (pulse_n // 2))

t = np.linspace(0, dt * pulse_n, pulse_n)
# gaussian modulated sine wave source
a = 1e-2
src[:pulse_n] = a * (np.sin(2*np.pi*f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(dtype_).squeeze()
# src[:pulse_n] = a * (np.exp(-((t - t0) / t_half)**2)).astype(dtype_).squeeze()

plt.figure()
plt.plot(src)

plt.figure()
frequency = np.arange(100e6, 20e9, 10e6)
Vinc = rfn.utils.dtft_f(src, frequency, 1 / dt)
plt.plot(frequency / 1e9, conv.db20_lin(Vinc))

# compute discrete Fourier transform
freq = np.linspace(0.1, 3, 1000) * 1e9
fs = 1 / dt
Xf = utils.dtft_f(src, freq, fs)
# normalize spectrum to one
Xfn = Xf / np.max(np.abs(Xf))

# plt.figure()
# plt.plot(freq, 20 * np.log10(np.abs(Xfn)))


#################
# FDTD Code
#################
# coefficient in front of the previous time values of E
Ca = (2 * epsilon - (sigma * dt)) / (2 * epsilon + (sigma * dt))
# coefficient in front of the difference terms of H
Cb = (2 * dt) / ((2 * epsilon + (sigma * dt))) 

# coefficient in front of the previous time values of H
# Da = (2 * u0 - (sigma_m * dt)) / (2 * u0 + (sigma_m * dt))
# coefficient in front of the difference terms of E
Db = (dt) / (mu)

# resistive load
r0 = 50
r0_z = r0 / ms_z
r_x = Nx-10

r_y = ms_y_mid
r_z = sub_z

Ca_ez = Ca.copy()
Cb_ez = Cb.copy()

####
Ca_r = Ca[r_x-1, r_y-1, 1]
Cb_r = Cb[r_x-1, r_y-1, 1]

rterm = dx_h[r_x-1] * dy_h[r_y-1] * 2 * r0
denom = 1 + (Cb_r * dz[1] / rterm)

Ca_ez_r = (Ca_r - (Cb_r * dz[1] / rterm)) / denom
Cb_ez_r = (Cb_r) / denom

# ez component uses the cell properties of the cell to it's left (r_x - 1)
Ca_ez[r_x-1, r_y-1, 1] = Ca_ez_r
Cb_ez[r_x-1, r_y-1, 1] = Cb_ez_r

# 0 ohm resistors to connect the lumped element to the trace
Ca_ez[r_x-1, r_y-1, 0] = -1
Cb_ez[r_x-1, r_y-1, 0] = 0

Ca_ez[r_x-1, r_y-1, 2] = -1
Cb_ez[r_x-1, r_y-1, 2] = 0
###

## add source resistor
r_srcx = 10

Ca_r = Ca[r_srcx-1, r_y-1, 1]
Cb_r = Cb[r_srcx-1, r_y-1, 1]

rterm = dx_h[r_srcx-1] * dy_h[r_y-1] * 2 * r0
denom = 1 + (Cb_r * dz[1] / rterm)

Ca_ez_r = (Ca_r - (Cb_r * dz[1] / rterm)) / denom
Cb_ez_r = (Cb_r) / denom

Ca_ez[r_srcx-1, r_y-1, 1] = Ca_ez_r
Cb_ez[r_srcx-1, r_y-1, 1] = Cb_ez_r
###

# 0 ohm resistors to connect the lumped element to the trace
Ca_ez[r_srcx-1, r_y-1, 0] = -1
Cb_ez[r_srcx-1, r_y-1, 0] = 0

Ca_ez[r_srcx-1, r_y-1, 2] = -1
Cb_ez[r_srcx-1, r_y-1, 2] = 0

# voltage sources
# coefficient in front of resistive voltage source term
Vs_a = (Cb_r / rterm) / denom

stime = time.time()
# loop over each time step
for n in range(Nt -  1):

    # grid starts at bottom left corner. A half-cell is placed in front of each component so all field values have 
    # the same number of values. The extra half cell components are not updated.

    # hx update
    # edges along x do not get updated
    hx[n+1, 1:-1, :, :] = (hx[n, 1:-1, :, :]) + Db[:-1, :, :] * (
        (np.diff(ey[n], axis=2) * dz_inv)[1:-1, :, :] - (np.diff(ez[n], axis=1) * dy_inv)[1:-1, :, :]
    )

    # hy update
    # edges along y do not get updated
    hy[n+1, :, 1:-1, :] = (hy[n, :, 1:-1, :]) + Db[:, :-1, :] * (
        (np.diff(ez[n], axis=0) * dx_inv)[:, 1:-1, :] - (np.diff(ex[n], axis=2) * dz_inv)[:, 1:-1, :]
    )

    # hz update
    # edges along z do not get updated
    hz[n+1, :, :, 1:-1] = (hz[n, :, :, 1:-1]) + Db[:, :, :-1] * (
        (np.diff(ex[n], axis=1) * dy_inv)[:, :, 1:-1] - (np.diff(ey[n], axis=0) * dx_inv)[:, :, 1:-1] 
    )

    # ex update
    # edges along y and z do not get updated
    ex[n+1, :, 1:-1, 1:-1] = (Ca[:, :-1, :-1] * ex[n, :, 1:-1, 1:-1]) + Cb[:, :-1, :-1] * (
        (np.diff(hz[n+1], axis=1) * dy_h_inv)[:, :, 1:-1] - (np.diff(hy[n+1], axis=2) * dz_h_inv)[:, 1:-1, :]
    )

    # ey update
    # edges along x and z do not get updated
    ey[n+1, 1:-1, :, 1:-1] = (Ca[:-1, :, :-1] * ey[n, 1:-1, :, 1:-1]) + Cb[:-1, :, :-1] * (
        (np.diff(hx[n+1], axis=2) * dz_h_inv)[1:-1, :, :] - (np.diff(hz[n+1], axis=0) * dx_h_inv)[:, :, 1:-1]
    )

    # ez update
    # edges along x and y do not get updated
    ez[n+1, 1:-1, 1:-1, :] = (Ca_ez[:-1, :-1, :] * ez[n, 1:-1, 1:-1, :]) + Cb_ez[:-1, :-1, :] * (
        (np.diff(hy[n+1], axis=0) * dx_h_inv)[:, 1:-1, :] - (np.diff(hx[n+1], axis=1) * dy_h_inv)[1:-1, :, :]
    )

    # PEC trace
    ex[n+1, ms_x, ms_y_ex, ms_z] = 0
    ey[n+1, ms_x, ms_y, ms_z] = 0

    # add resistive voltage source
    # Vs_a is already divided by sub_z, so we don't need to split the voltage among each ez component
    ez[n+1, r_srcx, r_y, 1] -= Vs_a * (src[n+1]) # book says src/2
    
    # ez[n+1, r_srcx, ms_y_mid, sub_z] -= Cb[r_srcx,  ms_y_mid, sub_z] * (1e3 * src[n+1])

print(f"done. Elapsed: {time.time() - stime: .3f}")

# voltage in lumped port
v1 = -ez[:, r_srcx, ms_y_mid, 1] * dz[1]

# current in lumped port, defined as leaving the port
c1 = (hy[:, r_srcx, ms_y_mid, 1] - hy[:, r_srcx - 1, ms_y_mid, 1] ) * dx_h[r_srcx - 1]
c2 = (-hx[:, r_srcx, ms_y_mid, 1] + hx[:, r_srcx, ms_y_mid - 1, 1] ) * dy_h[ms_y_mid - 1]
i1 = c1 + c2

b1 = (v1 - (i1 * 50)) / 2
a1 = (v1 + (i1 * 50)) / 2

# voltage in port 2
v2 = -ez[:, r_x, ms_y_mid, 1] * dz[1]

# current in port 2, defined as entering the port
c1 = (hy[:, r_x, ms_y_mid, 1] - hy[:, r_x - 1, ms_y_mid, 1] ) * dx_h[r_x - 1]
c2 = (-hx[:, r_x, ms_y_mid, 1] + hx[:, r_x, ms_y_mid - 1, 1] ) * dy_h[ms_y_mid - 1]
i2 = -(c1 + c2)

# no entering voltage wave from port 2, just evaluate the voltage across the resistor
p2 = i2 * 50

# plt.figure()
# plt.plot(b1)
# plt.plot(a1)

# plt.figure()
# plt.plot(a1)
# plt.plot(p2)

# plt.figure()
# plt.plot(v1)
# plt.plot(i1 * 50)

frequency = np.arange(5e9, 15e9, 10e6)
B1 = utils.dtft_f(b1, frequency, 1 / dt)
A1 = utils.dtft_f(a1, frequency, 1 / dt)

B2 = utils.dtft_f(p2, frequency, 1 / dt)

fig, ax = plt.subplots()
ax.plot(frequency  / 1e9, conv.db20_lin(B1 / A1))
ax.plot(frequency / 1e9, conv.db20_lin(B2 / A1))
ax.set_xlabel("Frequency [GHz]")
ax.set_ylim([-40, 1])
ax.margins(x=0)
ax.legend(["S11", "S21"], loc="upper left")
ax.set_ylabel("dB")
mplm.line_marker(x=10)
plt.show()



ez = ez[:, :Nx, :Ny, :Nz]

def generate_gif():
    g = pv.ImageData()

    grid =  np.ones(spatial_shape) * dmax
    g.dimensions = grid.shape
    g.spacing = (dmax, dmax, dmax)

    # Open a gif
    plotter = pv.Plotter(off_screen=True)

    trace_pnts = np.array([(10, Ny//2, ms_z), (Nx-10, Ny//2, ms_z), (Nx-10, Ny//2 + 5, ms_z)])
    trace = pv.Rectangle(trace_pnts * dmax)
    plotter.add_mesh(trace, opacity=0.5)


    data = 20 * np.log10(np.abs(ez[50]))

    vmin = -20
    vmax = 20
    data = np.clip(data, vmin, vmax)

    g.point_data['values'] = data.flatten(order="F")
    plotter.add_volume(
        g, cmap="jet", opacity="linear", scalars="values", clim=[vmin, vmax]
    )

    # data = 20 * np.log10(np.abs(ez[40]))
    # data = np.clip(data, -80, -20)

    # g.point_data["values"][:] = data.flatten(order="F")     # update in-place                 # mark as modified
    plotter.camera.zoom(1)
    plotter.render()    
    plotter.add_axes()
    plotter.add_bounding_box()
    plotter.camera_position = "xz"
    plotter.camera.elevation += 20
    plotter.camera.azimuth += 10
    plotter.camera.zoom(1.5)
    # plotter.show()


    plotter.open_gif('msline.gif')
    nstep = 15
    nframe = Nt // nstep
    for n in range(nframe):
        data = 20 * np.log10(np.abs(ez[n*nstep]))
        data = np.clip(data, vmin, vmax)
        g.point_data["values"][:] = data.flatten(order="F")
        plotter.render()  
        plotter.write_frame()
    # plotter.show()

    # Closes and finalizes movie
    plotter.close()

    


generate_gif()
ipyimage(filename='msline.gif')


