
import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils

u0 = const.u0
e0 = const.e0
c0 = const.c0

imax = 70
jmax = 80
kmax = 50
nmax = 300
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
    ez[n+1, imax//2, jmax//2, kmax//2] -= Cb[imax//2, jmax//2, kmax//2] * Jz_src[n]


fig, ax = plt.subplots()

im = ax.pcolormesh(20 * np.log10(np.abs(ez[200, :, :, kmax//2 - 10])), vmin=-80)
fig.colorbar(im)


fig, ax = plt.subplots()
ax.plot(ez[:, imax//2 + 20, jmax//2, kmax//2])
