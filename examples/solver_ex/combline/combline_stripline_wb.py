import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time
from scipy.special import ellipk
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

import rfnetwork as rfn
import mpl_markers as mplm
import sys

from scipy.optimize import least_squares

pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)

plt.style.use('ggplot')
# Set the font family to serif, with "Times New Roman" as the preferred serif font
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


sys.argv = sys.argv[0:1]

dir_ = Path(__file__).parent

u0 = const.u0
e0 = const.e0
c0 = const.c0
eta0 = const.eta0

er = 3.66

f1 = 1.5e9
f2 = 1.6e9

f0 = 1.5e9
w = 0.55

f1 = f0 - (w * f0)/ 2
f2 = f0 + (w * f0)/ 2

b = 0.06

g_wb = [1, 1.1897, 1.4346, 2.1199, 1.6010, 2.1699, 1.5640, 1.9444, 0.8778, 1.3554]
g_nb = [1, 1.1681, 1.4039, 2.0562, 1.5170, 1.9029, 0.8618, 1.3554]

g = [1, 0.5176, 1.4142, 1.9318, 1.9318, 1.4142, 0.5176, 1.0000]
Ck, Cmk = rfn.utils.combline_sections_wb(g, f1, f2, er=er, h=0.2)

# print(Ck, Cmk)

def find_Cab_spacing(sp, target_Cab):
   w = b * 0.5
   Cf_o, Cf_e = utils.coupled_sline_fringing_cap(w, sp, b, er)
   # normalized Cab
   Cab = (Cf_o - Cf_e) / (e0 * er)

   return Cab - target_Cab

# determine spacings using the capacitance between lines Cmk
sk = np.zeros_like(Cmk)
for i, cmk in enumerate(Cmk):
    sk[i] = least_squares(find_Cab_spacing, x0=b*0.5, args=(cmk,), bounds=(0.001, b)).x[0]

# even mode fringing capacitance for each space between lines, width is arbitrary here
Cf_e = np.array([utils.coupled_sline_fringing_cap(0.5*b, s, b, er)[1] for s in sk]) / (e0 * er)
# fringing capacitance on the outer edges (not between the two lines), figure 5.05-10b, for t=0
t = 0
# Cf = (2 / np.pi) * np.log((1 / (1 - t/b) + 1)) - (t / (np.pi * b)) * np.log((1 / (1 - t/b)**2 - 1))
Cf = 0.44
# Ck is the even mode capacitance Ce. Use equation 5.05-25 to determine the per unit length parallel plate capacitance 
# for each line. Normalized by eps
Cf_eps_left = np.concatenate([[Cf], Cf_e])
Cf_eps_right = np.concatenate([Cf_e, [Cf]])
Cp_e = (Ck / 2) - Cf_eps_left - Cf_eps_right
# determine width using parallel plate capacitance Cp_e = 2w / b
wk = (Cp_e / 2) * b

print(wk, sk)

# wk[0] -= 0.005
# wk[-1] -= 0.005
# fringing_capacitance_figure(b=b, w=0.3, er=er)

# Cf = np.sqrt(er) / rfn.const.c0 * 

# design taken from table 10.07-2 in Matthaei
K = len(wk)
#k       0      1      2      3      4      5      6      7
# w_k =   [0.126, 0.121, 0.126, 0.127, 0.127, 0.126, 0.121, 0.126]
# s_k =   [0.092, 0.136, 0.143, 0.146, 0.143, 0.136, 0.087]

# y coordinates of the bottom and top edge of each line
ymax = rfn.const.c0_in / (f0 * np.sqrt(er) * 4)
y0 = 0.05 * ymax
y1 = ymax - y0

# x coordinates of the left and right edge of each line
x0_k = np.zeros(K)
x1_k = np.zeros(K)
y0_k = np.zeros(K)
y1_k = np.zeros(K)

x0_k[0] = 0.15
x1_k[0] = x0_k[0] + wk[0]

for i in range(K):
    if i % 2: # if odd
        y0_k[i] = y0
        y1_k[i] = ymax
    else: # if even
        y0_k[i] = 0
        y1_k[i] = y1

y0_k[0] = 0.025 * ymax
y1_k[-1] = ymax - (0.025 * ymax)

for i in range(1, K):
    x0_k[i] = x1_k[i-1] + sk[i-1]
    x1_k[i] = x0_k[i] + wk[i]


sbox_w = x1_k[-1] + 0.15
sbox_len = ymax
sbox_h = b
substrate = pv.Cube(center=(sbox_w/2, sbox_len/2, 0), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)
sbox =      pv.Cube(center=(sbox_w/2, sbox_len/2, 0), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)

s = rfn.Solver_PCB(sbox, nports=4)
s.add_substrate("sub", substrate, er=er, opacity=0.0, loss_tan=0.0035, f0=1.5e9)


for i in range(K):
    line = pv.Rectangle([
        (x0_k[i], y0_k[i], 0),
        (x1_k[i], y0_k[i], 0),
        (x1_k[i], y1_k[i], 0),
    ])

    s.add_pec_face(f"line_{i}", line, color="gold")

port1_face = pv.Rectangle([
    (x0_k[0], y0_k[0], 0),
    (x1_k[0], y0_k[0], 0),
    (x1_k[0], y0_k[0], -sbox_h/2),
])

port2_face = pv.Rectangle([
    (x0_k[0], y0_k[0], 0),
    (x1_k[0], y0_k[0], 0),
    (x1_k[0], y0_k[0], sbox_h/2),
])

port34_y = y0_k[-1] if y1_k[-1] == ymax else y1_k[-1]

port3_face = pv.Rectangle([
    (x0_k[-1], port34_y, 0),
    (x1_k[-1], port34_y, 0),
    (x1_k[-1], port34_y, -sbox_h/2),
])


port4_face = pv.Rectangle([
    (x0_k[-1], port34_y, 0),
    (x1_k[-1], port34_y, 0),
    (x1_k[-1], port34_y, sbox_h/2),
])

s.add_lumped_port(1, port1_face)
s.add_lumped_port(2, port2_face)
s.add_lumped_port(3, port3_face)
s.add_lumped_port(4, port4_face)


# s.init_mesh(d0 = lam0/20, n0 = 2, d_pec = lam0/20, n_min_pec=4, d_sub=lam0/20, n_min_sub=4, blend_pec=True)
s.init_mesh_edge_method(d0 = 0.05, d_edge = 0.0025)
s.init_coefficients()

# s.init_mesh_edge_method(d0 = 0.1, d_edge=0.01)
# s.init_coefficients()

self = s

plotter = s.render()
plotter.camera_position = "xy"
plotter.show()
print(s.Nx * s.Ny * s.Nz / 1e3, "kcells")


s.init_ports(r0=100)
s.init_pec()

s.add_field_monitor("mon1", "ez", "z", 0, 30)
# s.add_field_monitor("mon1", "ey", "z", sub_h, 5)
# s.add_field_monitor("mon2", "ey", "z", sub_h, 15)
# s.add_field_monitor("mon3", "ex", "z", sub_h, 10)

pulse_n = 150000
# # width of half pulse in time
# t_half = (s.dt * 100)
# # center of the pulse in time
# t0 = (s.dt * 400)

vsrc = 1e-2 * s.gaussian_source(s.dt * 300, s.dt * pulse_n)

# t = np.linspace(0, s.dt * pulse_n, pulse_n)
# vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

frequency: np.ndarray = np.arange(0.4e9, 6e9, 2e6)

s.run([1, 2], [vsrc, -vsrc], n_threads=4)
self = s


p = s.plot_monitor(
    ["mon1"], zoom=1.1, view="xy", el=0, opacity=[0.9, 1], 
    linear=False, cmap="jet", style="surface",
)
# p.camera_position = "xy"
p.show(title="EM Solver")


sdata = s.get_sparameters(frequency, 1, z0=100)
S11 = sdata[:, 0]
S21 = sdata[:, 2]


fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0,0]
rfn.plots.draw_smithchart(ax)
ax.plot(S11.real, S11.imag)

ax = axes[1,1]
ax.plot(frequency / 1e9, conv.db20_lin(S11))
ax.set_xlabel("Frequency [GHz]")
ax.set_xticks(np.arange(0.4, 3.2, 0.2))
ax.set_xlim([0.4, 3])
ax.set_ylabel("[dB]")
ax.set_ylim([-40, 2])
ax.grid(True)
ax.legend(["S11"], loc="lower left")

ax = axes[1,0]
ax.plot(frequency / 1e9, conv.db20_lin(S21))
mplm.line_marker(x = f0/1e9, axes=ax)
ax.set_xticks(np.arange(0.4, 3.2, 0.2))
ax.set_xlim([0.4, 3])
ax.set_ylim([-60, 2])
ax.grid(True)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S21"], loc="lower left")

ax = axes[0,1]
ax.plot(frequency / 1e9, np.unwrap(np.angle(S21, deg=True)))
ax.set_xticks(np.arange(0.4, 3.2, 0.2))
ax.set_xlim([0.4, 3])
mplm.line_marker(x = f0/1e9, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[deg]")
ax.legend(["S21"], loc="lower left")

fig.tight_layout()
plt.show()