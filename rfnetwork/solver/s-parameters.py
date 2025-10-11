
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
    w=0.04, 
    h=0.020, 
    er=3.66, 
)

m = msline50(2)

# 1. too wide by 30mils (1.5 cells)
# 2. sub too thin by 8 mils 
frequency = np.arange(5e9, 15e9, 10e6)
# m.plot(11, frequency=frequency, fmt="smith")
# plt.ylim([-40, 0])


msline50.get_properties(10e9)

# number of cells in each dimension

Nt = 2500
f0 = 10e9
fmax = 15e9

max_er: float = 3.66
# minimum number of cells per wavelength
cells_per_wavelength: int = 15
dtype_ = np.float32

# smallest wavelength
vp = c0 / np.sqrt(max_er)
lam_min = vp / fmax

# largest allowed cell length
dmax = conv.m_in(0.02) #lam_min / cells_per_wavelength
conv.in_m(dmax)

Nx = int((2 / conv.in_m(dmax)) + 20)
Ny = 20
Nz = 25
spatial_shape = (Nx, Ny, Nz)

# field values
ex = np.zeros((Nt, Nx, Ny+1, Nz+1), dtype=dtype_)
ey = np.zeros((Nt, Nx+1, Ny, Nz+1), dtype=dtype_)
ez = np.zeros((Nt, Nx+1, Ny+1, Nz), dtype=dtype_)
hx = np.zeros((Nt, Nx+1, Ny, Nz), dtype=dtype_)
hy = np.zeros((Nt, Nx, Ny+1, Nz), dtype=dtype_)
hz = np.zeros((Nt, Nx, Ny, Nz+1), dtype=dtype_)

# cell sizes
dx = np.ones(Nx) * (dmax)
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
ms_y = slice(y_mid, y_mid + 2)
ms_y_ex = slice(y_mid, y_mid + 3)
ms_x = slice(10, Nx -10)
ms_z = 2
sub_z = slice(1, ms_z)

w = 0.040
h = 0.020
sub_er = 3.66

# trace
dy[ms_y] = conv.m_in(w / 2)
# dz[5] = conv.m_in(cu_h)
# sigma[10:-10, ms_y, 5] = 6e7

# substrate
dz[sub_z] = conv.m_in(h / 1)
# epsilon[:, :, sub_z] = sub_er * e0
# (epsilon / e0)[0, 0]
# dz[sub_z.stop: sub_z.stop + 3] = (dz[ms_z] + dz[ms_z - 1]) / 2

# compute maximum time step that ensures convergence, use freespace propagation speed as worst case
length_min = np.array([np.min(dx), np.min(dy), np.min(dz)])
dmin = 1 / np.sqrt(((1 / length_min)**2).sum())
# S = 0.80 * (1 / np.sqrt(3))
dt = 0.95 * (dmin / const.c0)
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
t_half = 9e-11#(dt * (pulse_n // 8))
# center of the pulse in time
t0 = (dt * (pulse_n // 2))

t = np.linspace(0, dt * pulse_n, pulse_n)
# gaussian modulated sine wave source
a = 1e-2
src[:pulse_n] = a * (np.sin(2*np.pi*f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(dtype_).squeeze()
# src[:pulse_n] = a * (np.exp(-((t - t0) / t_half)**2)).astype(dtype_).squeeze()

# plt.figure()
# plt.plot(src)

# plt.figure()
frequency = np.arange(100e6, 20e9, 10e6)
Vinc = rfn.utils.dtft_f(src, frequency, 1 / dt)
# plt.plot(frequency / 1e9, conv.db20_lin(Vinc))

#################
# Materials
#################
sig_0 = 0
Ca_0 = (2 * e0 - (sig_0 * dt)) / (2 * e0 + (sig_0 * dt))
Cb_0 = (2 * dt) / ((2 * e0 + (sig_0 * dt))) 
# Da = (2 * u0 - (sigma_m * dt)) / (2 * u0 + (sigma_m * dt))
Db_0 = (dt) / (u0)

# coefficient in front of the previous time values of E
Ca_ex = np.ones(ex[0].shape) * Ca_0
Ca_ey = np.ones(ey[0].shape) * Ca_0
Ca_ez = np.ones(ez[0].shape) * Ca_0

# coefficient in front of the difference terms of H
Cb_ex = np.ones(ex[0].shape) * Cb_0
Cb_ey = np.ones(ey[0].shape) * Cb_0
Cb_ez = np.ones(ez[0].shape) * Cb_0

# coefficient in front of the difference terms of E
Db_hx = np.ones(hx[0].shape) * Db_0
Db_hy = np.ones(hy[0].shape) * Db_0
Db_hz = np.ones(hz[0].shape) * Db_0

# substrate 
sub_eps = 3.6 * e0
Cb_ez[..., sub_z] = (2 * dt) / ((2 * sub_eps + (sig_0 * dt))) 
Cb_ex[..., sub_z] = (2 * dt) / ((2 * sub_eps + (sig_0 * dt))) 
Cb_ey[..., sub_z] = (2 * dt) / ((2 * sub_eps + (sig_0 * dt))) 

# ex and ey are on the boundary between material cells, compute the average of the substrate
# FIX for non-uniform grids
Cb_ex[..., ms_z] = (Cb_ex[..., ms_z-1] + Cb_ex[..., ms_z + 1]) / 2
Cb_ey[..., ms_z] = (Cb_ey[..., ms_z-1] + Cb_ey[..., ms_z + 1]) / 2


# PEC trace
# ex/ey on trace
sig_pec = 1e6
er_eff = (sub_eps + e0) / 2
Ca_ex[ms_x, ms_y.start: ms_y.stop+1, ms_z] = (2 * er_eff - (sig_pec * dt)) / (2 * er_eff + (sig_pec * dt))
Ca_ey[ms_x.start: ms_x.stop + 1, ms_y, ms_z] = (2 * er_eff - (sig_pec * dt)) / (2 * er_eff + (sig_pec * dt))

Cb_ex[ms_x, ms_y.start: ms_y.stop+1, ms_z] = (2 * dt) / ((2 * er_eff + (sig_pec * dt))) 
Cb_ey[ms_x.start: ms_x.stop + 1, ms_y, ms_z] = (2 * dt) / ((2 * er_eff + (sig_pec * dt))) 

# gnd plane
Ca_ex[..., 1] = (2 * e0 - (sig_pec * dt)) / (2 * e0 + (sig_pec * dt))
Ca_ey[..., 1] = (2 * e0 - (sig_pec * dt)) / (2 * e0 + (sig_pec * dt))

Cb_ex[..., 1] = (2 * dt) / ((2 * e0 + (sig_pec * dt))) 
Cb_ey[..., 1] = (2 * dt) / ((2 * e0 + (sig_pec * dt))) 


# resistive load
r0 = 50
r_x = Nx-10

r_y = ms_y_mid 
r_z = 1

# ez resistor properties
Ca_r = Ca_ez[r_x, r_y, r_z]
Cb_r = Cb_ez[r_x, r_y, r_z]

rterm = (r0 * dy[0] *dx[0])

denom = (sub_eps / dt) + (dz[0] / (2 * rterm))

# ez component for resistor
Ca_ez[r_x, r_y, r_z] = ((sub_eps / dt) - (dz[0] / (2 * rterm))) / denom
Cb_ez[r_x, r_y, r_z] = 1 / (denom)

# 0 ohm resistors to connect the lumped element to the trace
# Ca_ez[r_x-1, r_y-1, 0] = -1
# Cb_ez[r_x-1, r_y-1, 0] = 0

# Ca_ez[r_x-1, r_y-1, 2] = -1
# Cb_ez[r_x-1, r_y-1, 2] = 0
###

## add source resistor
r_srcx = 10

Ca_ez[r_srcx, r_y, r_z] = ((sub_eps / dt) - (dz[0] / (2 * rterm))) / denom
Cb_ez[r_srcx, r_y, r_z] = 1 / (denom)

# voltage sources
# coefficient in front of resistive voltage source term
Vs_a = 1 / (denom * rterm)


# 0 ohm resistors to connect the lumped element to the trace
# Ca_ez[r_srcx-1, r_y-1, 0] = -1
# Cb_ez[r_srcx-1, r_y-1, 0] = 0

# Ca_ez[r_srcx-1, r_y-1, 2] = -1
# Cb_ez[r_srcx-1, r_y-1, 2] = 0
###

# h components around load, thin wire model
a = conv.m_in(1e-9)

Db_hz_ex = Db_hz.copy()
Db_hz_ey = Db_hz.copy()

Db_hx_ey = Db_hx.copy()
Db_hx_ez = Db_hx.copy()

Db_hy_ez = Db_hy.copy()
Db_hy_ex = Db_hy.copy()

# shorten load along the x direction
u0_c1 = u0 * 0.75
Db_hy_ez[r_srcx - 1, r_y, r_z] = (2 * dt) / (u0_c1 * np.log(dx[0] / a))
Db_hy_ez[r_x, r_y, r_z] = (2 * dt) / (u0_c1 * np.log(dx[0] / a))

# shorten the load inside the trace
# Db_hy_ez[r_srcx, r_y, r_z] = (2 * dt) / (u0_c1 * np.log(dx[0] / a))
# Db_hy_ez[r_x-1, r_y, r_z] = (2 * dt) / (u0_c1 * np.log(dx[0] / a))

# # shorten the load along the y direction
# Db_hx_ez[r_srcx, r_y, r_z] = (2 * dt) / (u0_c1 * np.log(dy[0] / a))
# Db_hx_ez[r_srcx, r_y-1, r_z] = (2 * dt) / (u0_c1 * np.log(dy[0] / a))

# Db_hx_ez[r_x, r_y, r_z] = (2 * dt) / (u0_c1 * np.log(dy[0] / a))
# Db_hx_ez[r_x, r_y-1, r_z] = (2 * dt) / (u0_c1 * np.log(dy[0] / a))


# inductance compensation
u0_cy = u0 * 0.75
Db_hy_ez[r_srcx, r_y, r_z] = (dt) / (u0_cy)
Db_hy_ez[r_x-1, r_y, r_z] = (dt) / (u0_cy)

# hx 
u0_cx = u0 * 0.75
Db_hx_ez[r_srcx, r_y, r_z] = (dt) / (u0_cx)
Db_hx_ez[r_srcx, r_y-1, r_z] = (dt) / (u0_cx)

Db_hx_ez[r_x, r_y, r_z] = (dt) / (u0_cx)
Db_hx_ez[r_x, r_y-1, r_z] = (dt) / (u0_cx)



# ke = 1 / ((dx[0] / dy[0]) * np.arctan(dy[0] / dx[0]))
# Db_hz_ey[ms_x, ms_y.start - 1, ms_z] = ke * dt  / ( u0 )
# Db_hz_ey[ms_x, ms_y.stop + 1, ms_z] = ke * dt  / ( u0 )

# Db_hz_ex[ms_x.start, ms_y, 1] = (2 * dt) / (u0 * np.log(dx[0] / a))
# Db_hz_ex[ms_x.stop+1, ms_y, 1] = (2 * dt) / (u0 * np.log(dx[0] / a))

#################
# FDTD Code
#################

stime = time.time()
# loop over each time step
for n in range(Nt -  1):

    # grid starts at bottom left corner. A half-cell is placed in front of each component so all field values have 
    # the same number of values. The extra half cell components are not updated.

    # ex update
    # edges along y and z do not get updated
    ex[n+1, :, 1:-1, 1:-1] = (Ca_ex[:, 1:-1, 1:-1] * ex[n, :, 1:-1, 1:-1]) + Cb_ex[:, 1:-1, 1:-1] * (
        (np.diff(hz[n], axis=1) * dy_h_inv)[:, :, 1:-1] - (np.diff(hy[n], axis=2) * dz_h_inv)[:, 1:-1, :]
    )

    # ey update
    # edges along x and z do not get updated
    ey[n+1, 1:-1, :, 1:-1] = (Ca_ey[1:-1, :, 1:-1] * ey[n, 1:-1, :, 1:-1]) + Cb_ey[1:-1, :, 1:-1] * (
        (np.diff(hx[n], axis=2) * dz_h_inv)[1:-1, :, :] - (np.diff(hz[n], axis=0) * dx_h_inv)[:, :, 1:-1]
    )

    # ez update
    # edges along x and y do not get updated
    ez[n+1, 1:-1, 1:-1, :] = (Ca_ez[1:-1, 1:-1, :] * ez[n, 1:-1, 1:-1, :]) + Cb_ez[1:-1, 1:-1, :] * (
        (np.diff(hy[n], axis=0) * dx_h_inv)[:, 1:-1, :] - (np.diff(hx[n], axis=1) * dy_h_inv)[1:-1, :, :]
    )

    # # PEC trace
    # ex[n+1, ms_x.start: ms_x.stop -1, ms_y_ex, ms_z] = 0
    # ey[n+1, ms_x, ms_y, ms_z] = 0


    # add resistive voltage source
    # Vs_a is already divided by sub_z, so we don't need to split the voltage among each ez component
    ez[n+1, r_srcx, r_y, r_z] -= Vs_a * (src[n+1]) # book says src/2

    # hx update
    # all edges are updated
    hx[n+1, :, :, :] = (hx[n, :, :, :]) + (
        Db_hx_ey * (np.diff(ey[n+1], axis=2) * dz_inv) - Db_hx_ez * (np.diff(ez[n+1], axis=1) * dy_inv)
    )

    # hy update
    # all edges are updated
    hy[n+1, :, :, :] = (hy[n, :, :, :]) + (
        Db_hy_ez * (np.diff(ez[n+1], axis=0) * dx_inv) - Db_hy_ex * (np.diff(ex[n+1], axis=2) * dz_inv)
    )

    # hz update
    # all edges are updated
    hz[n+1, :, :, :] = (hz[n, :, :, :]) + (
        Db_hz_ex * (np.diff(ex[n+1], axis=1) * dy_inv) - Db_hz_ey * (np.diff(ey[n+1], axis=0) * dx_inv)
    )

    # ez[n+1, r_srcx, ms_y_mid, sub_z] -= Cb[r_srcx,  ms_y_mid, sub_z] * (1e3 * src[n+1])

print(f"done. Elapsed: {time.time() - stime: .3f}")

# current on trace (in middle)
x_probe = ms_x.start + 10
im = np.sum(
    hy[:, x_probe, ms_y.start: ms_y.stop + 1, ms_z-1] * dy_h[ms_y.start: ms_y.stop + 1][None] -
    hy[:, x_probe, ms_y.start: ms_y.stop + 1, ms_z] * dy_h[ms_y.start: ms_y.stop + 1][None], axis=-1
)

im += (-hz[:, x_probe, ms_y.start-1, ms_z] * dz_h[1]) + (hz[:, x_probe, ms_y.stop+1, ms_z] * dz_h[1])

# voltage across trace conductors
vm = -((ez[:, x_probe, ms_y_mid, r_z] + ez[:, (x_probe) + 1, ms_y_mid, r_z]) / 2) * dz[r_z]

# plt.figure()
# plt.plot(vm)

frequency = np.arange(5e9, 15e9, 10e6)
Vm = utils.dtft_f(vm[:900], frequency, 1 / dt)
Im = utils.dtft_f(im[:900], frequency, 1 / dt) * np.exp(1j * 2 * np.pi * frequency * dt / 2)

fig, ax = plt.subplots()
ax.plot(frequency  / 1e9, Vm / Im)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("$Z_{fwd}$")

fig, ax = plt.subplots()
ax.plot(frequency  / 1e9, conv.db20_lin(conv.gamma_z(Vm / Im)))


# voltage in lumped port
v1 = -ez[:, r_srcx, ms_y_mid, r_z] * dz[r_z]

# current in lumped port, defined as leaving the port
c1 = (hy[:, r_srcx, ms_y_mid, r_z] - hy[:, r_srcx - 1, ms_y_mid, r_z] ) * dx_h[r_srcx - 1]
c2 = (-hx[:, r_srcx, ms_y_mid, r_z] + hx[:, r_srcx, ms_y_mid - 1, r_z] ) * dy_h[ms_y_mid - 1]
i1 = c1 + c2


frequency = np.arange(5e9, 15e9, 10e6)
V1 = utils.dtft_f(v1, frequency, 1 / dt)
I1 = utils.dtft_f(i1, frequency, 1 / dt)

# delay current by half a time-step to be at the same time sample as the voltage
# h components are ahead of the e components by half a time step
I1 = I1 * np.exp(1j * 2 * np.pi * frequency * dt / 2)

z0 = 50 #+ 1j * 2 * np.pi * 10e9 * 0.05e-9
A1 = (V1 + z0 * I1) / (2 * np.sqrt(z0.real))
B1 = (V1 - np.conj(z0) * I1) / (2 * np.sqrt(z0.real))

# fig, ax = plt.subplots()
# ax.plot(frequency  / 1e9, conv.db20_lin(B1 / A1))
# mplm.line_marker(x=10)
# ax.set_ylim([-40, 1])


b1 = (v1 - (i1 * 50)) / 2
a1 = (v1 + (i1 * 50)) / 2

# voltage in port 2
v2 = -ez[:, r_x, ms_y_mid, r_z] * dz[r_z]

# current in port 2, defined as entering the port
c1 = (hy[:, r_x, ms_y_mid, r_z] - hy[:, r_x - 1, ms_y_mid, r_z] ) * dx_h[r_x - 1]
c2 = (-hx[:, r_x, ms_y_mid, r_z] + hx[:, r_x, ms_y_mid - 1, r_z] ) * dy_h[ms_y_mid - 1]
i2 = -(c1 + c2)

# evaluate the voltage across the resistor
frequency = np.arange(5e9, 15e9, 10e6)
V2 = utils.dtft_f(v2, frequency, 1 / dt)
I2 = utils.dtft_f(i2, frequency, 1 / dt)

# delay current by half a time-step to be at the same time sample as the voltage
# h components are ahead of the e components by half a time step
I2 = I2 * np.exp(1j * 2 * np.pi * frequency * dt / 2)

z0 = 50 #+ 1j * 2 * np.pi * 10e9 * 0.05e-9
B2 = (V2 + z0 * I2) / (2 * np.sqrt(z0.real))
# B2 = (V2 - np.conj(z0) * I2) / (2 * np.sqrt(z0.real))

plt.figure()
plt.plot(b1)
plt.plot(a1)

fig, ax = plt.subplots()
ax_t = ax.twinx()
ax.plot(frequency  / 1e9, conv.db20_lin(B1 / A1))
ax_t.plot(frequency / 1e9, conv.db20_lin(B2 / A1), color="orange", alpha=0.5)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylim([-50, 1])
ax.margins(x=0)
ax.legend(["S11", "S21"], loc="upper left")
ax_t.legend(["S21"], loc="upper right")
ax.set_ylabel("dB")
ax_t.set_ylabel("dB")
mplm.line_marker(x=10)
plt.show()

S11 = B1 / A1
fig, ax = plt.subplots()
rfn.plots.draw_smithchart(ax)
plt.plot(S11.real, S11.imag)

# impedance plot
# fig, ax = plt.subplots()
# ax.plot(frequency, np.real(conv.z_gamma(S11, refz=50)))


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
    vmax = 30
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


