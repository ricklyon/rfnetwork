
import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv

from IPython.display import Image as ipyimage
from PIL import Image
import io

u0 = const.u0
e0 = const.e0
c0 = const.c0

imax = 100
jmax = 100
kmax = 100
nmax = 150
f0 = 1.5e9
fmax = 3e9

spatial_shape = (imax, jmax, kmax)

max_er: float = 1.5,
# minimum number of cells per wavelength
cells_per_wavelength: int = 20
dtype_ = np.float32

# smallest wavelength
vp = c0 / np.sqrt(max_er)
lam_min = vp / fmax

# largest allowed spatial step size
del_max = lam_min / cells_per_wavelength

# compute maximum time step that ensures convergence, use freespace propagation speed as worst case
S = 0.80 * (1 / np.sqrt(3))
dt = S * (del_max / const.c0)

# field values, includes an extra half-cell at the beginning of each axis so all fields have the same 
# number of components
ex = np.zeros((nmax,) + spatial_shape, dtype=dtype_)
ey = np.zeros((nmax,) + spatial_shape, dtype=dtype_)
ez = np.zeros((nmax,) + spatial_shape, dtype=dtype_)
hx = np.zeros((nmax,) + spatial_shape, dtype=dtype_)
hy = np.zeros((nmax,) + spatial_shape, dtype=dtype_)
hz = np.zeros((nmax,) + spatial_shape, dtype=dtype_)

# material properties
epsilon = np.ones(spatial_shape, dtype=dtype_) * e0
mu = np.ones(spatial_shape, dtype=dtype_) * u0
mu_m = np.zeros(spatial_shape, dtype=dtype_)
sigma = np.zeros(spatial_shape, dtype=dtype_)
sigma_m = np.zeros(spatial_shape, dtype=dtype_)

# cell sizes
del_x = np.ones(imax - 1)[:, None, None] * (1 / del_max)
del_y = np.ones(jmax - 1)[None, :, None] * (1 / del_max)
del_z = np.ones(kmax - 1)[None, None, :] * (1 / del_max)

# source
# width of half pulse in time
t_half = (dt * (nmax // 4))
# center of the pulse in time
t0 = (dt * (nmax // 2))

t = np.linspace(0, dt * nmax, nmax)
# gaussian modulated sine wave source
a = 1
Jz_src = a * (np.sin(2*np.pi*f0 * (t - t0)) * np.exp(-((t - t0) / t_half)**2)).astype(dtype_).squeeze()

plt.figure()
plt.plot(Jz_src)

# compute discrete Fourier transform
freq = np.linspace(0.1, 3, 1000) * 1e9
fs = 1 / dt
Xf = utils.dtft_f(Jz_src, freq, fs)
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
Da = (2 * u0 - (sigma_m * dt)) / (2 * u0 + (sigma_m * dt))
# coefficient in front of the difference terms of E
Db = (2 * dt) / ((2 * u0 + (sigma_m * dt)))


# loop over each time step
for n in range(nmax-1):

    # grid starts at bottom left corner. A half-cell is placed in front of each component so all field values have 
    # the same number of values. The extra half cell components are not updated.

    # hx update
    # first component along y and z are padding and do not get updated
    hx[n+1, :, 1:, 1:] = (Da[:, 1:, 1:] * hx[n, :, 1:, 1:]) + Db[:, 1:, 1:] * (
        (np.diff(ey[n], axis=2) * del_z)[:, 1:, :] - (np.diff(ez[n], axis=1) * del_y)[:, :, 1:]
    )

    # hy update
    # first component along x and z are padding and do not get updated
    hy[n+1, 1:, :, 1:] = (Da[1:, :, 1:] * hy[n, 1:, :, 1:]) + Db[1:, :, 1:] * (
        (np.diff(ez[n], axis=0) * del_x)[:, :, 1:] - (np.diff(ex[n], axis=2) * del_z)[1:, :, :]
    )

    # hz update
    # first component along x and y are padding and do not get updated
    hz[n+1, 1:, 1:, :] = (Da[1:, 1:, :] * hz[n, 1:, 1:, :]) + Db[1:, 1:, :] * (
        (np.diff(ex[n], axis=1) * del_y)[1:, :, :] - (np.diff(ey[n], axis=0) * del_x)[:, 1:, :]
    )

    # ex update
    # last components along y and z have no h component next to them and do not get updated
    ex[n+1, :, :-1, :-1] = (Ca[:, :-1, :-1] * ex[n, :, :-1, :-1]) + Cb[:, :-1, :-1] * (
        (np.diff(hz[n+1], axis=1) * del_y)[:, :, :-1] - (np.diff(hy[n+1], axis=2) * del_z)[:, :-1, :]
    )

    # ey update
    # last components along x and z have no h component next to them and do not get updated
    ey[n+1, :-1, :, :-1] = (Ca[:-1, :, :-1] * ey[n, :-1, :, :-1]) + Cb[:-1, :, :-1] * (
        (np.diff(hx[n+1], axis=2) * del_z)[:-1, :, :] - (np.diff(hz[n+1], axis=0) * del_x)[:, :, :-1]
    )

    # ez update
    # last components along x and y have no h component next to them and do not get updated
    ez[n+1, :-1, :-1, :] = (Ca[:-1, :-1, :] * ez[n, :-1, :-1, :]) + Cb[:-1, :-1, :] * (
        (np.diff(hy[n+1], axis=0) * del_x)[:, :-1, :] - (np.diff(hx[n+1], axis=1) * del_y)[:-1, :, :]
    )

    # add current sources
    ez[n+1, imax//2, jmax//2, kmax//2 -2:kmax//2 + 2] -= Cb[imax//2, jmax//2, kmax//2] * Jz_src[n]




# fig, ax = plt.subplots()
# ax.plot(ez[:, imax//2 + 20, jmax//2, kmax//2])


# grid =  np.ones(spatial_shape) * del_max



# data = 20 * np.log10(np.abs(ez[200]))
# data = np.clip(data, -80, -20)


# g.point_data['values'] = data.flatten(order="F")
# g.point_data["values"] = np.linspace(0, 10, np.prod(spatial_shape)).reshape(spatial_shape).flatten()
# g.plot(volume=True, cmap="jet", opacity="sigmoid")

g = pv.ImageData()

grid =  np.ones(spatial_shape) * del_max
g.dimensions = grid.shape
g.spacing = (del_max, del_max, del_max)

# Open a gif
plotter = pv.Plotter(off_screen=True)

data = 20 * np.log10(np.abs(ez[50]))

vmin = -55
vmax = -20
data = np.clip(data, vmin, vmax)

g.point_data['values'] = data.flatten(order="F")
plotter.add_volume(
    g, cmap="jet", opacity="linear", scalars="values", clim=[vmin, vmax]
)

# data = 20 * np.log10(np.abs(ez[40]))
# data = np.clip(data, -80, -20)

# g.point_data["values"][:] = data.flatten(order="F")     # update in-place                 # mark as modified
plotter.camera.zoom(2)
plotter.render()    


plotter.open_gif('wave.gif')
nframe = nmax // 3
for n in range(nframe):
    data = 20 * np.log10(np.abs(ez[n*3]))
    data = np.clip(data, vmin, vmax)
    g.point_data["values"][:] = data.flatten(order="F")
    plotter.render()  
    plotter.write_frame()
# plotter.show()

# Closes and finalizes movie
plotter.close()

ipyimage(filename='wave.gif')

# data = 20 * np.log10(np.abs(ez[200]))
# data = np.clip(data, -80, -20)
# g['values'] = data.flatten(order="F")
# plotter.update_scalars(data.flatten(order="F"), mesh=g, render=True)

# plotter.show()

# fig, ax = plt.subplots()
# ax.set_axis_off()
# images = []

# # plotter.open_gif('wave.gif')

# # Update Z and write a frame for each updated position
# nframe = nmax // 2
# for n in range(nframe):
#     # Update values inplace
#     data = 20 * np.log10(np.abs(ez[n*2]))
#     data = np.clip(data, -80, -20)
#     g.point_data["values"][:] = data.flatten(order="F")
#     plotter.render()  

#     # fig, ax = plt.subplots()
#     img = plotter.screenshot()
#     ax.imshow(img)
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png")
#     images.append(Image.open(buf))
#     # plt.close("all")
#     # Write a frame. This triggers a render.
#     # plotter.render()
    
#     # plotter.update_scalars(data.flatten(order="F"), mesh=g, render=True)
#     # plotter.write_frame()

# # # Closes and finalizes movie
# # plotter.close()

# gifname = f"mw_comparison.gif"
# images[0].save(
#     gifname,
#     format="GIF",
#     append_images=images,
#     save_all=True,
#     duration=50,
#     loop=0,
#     optimize=False,
# )




# fig, ax = plt.subplots()

# im = ax.pcolormesh(data[:, :, kmax//2])
# fig.colorbar(im)


# values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
# values.shape

# # Create the spatial reference
# grid = pv.ImageData()

# # Set the grid dimensions: shape because we want to inject our values on the
# #   POINT data
# grid.dimensions = values.shape

# # Edit the spatial reference
# grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
# grid.spacing = (1, 5, 2)  # These are the cell sizes along each axis

# # Add the data values to the cell data
# grid.point_data['values'] = values.flatten(order='F')  # Flatten the array

# # Now plot the grid
# grid.plot(volume=True, cmap="jet")