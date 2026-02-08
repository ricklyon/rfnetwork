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

f1 = 1.1e9
f2 = 1.6e9

f0 = (f1 + f2) / 2

# f0 = 1.4e9
# w = 0.2

# f1 = f0 - (w * f0)/ 2
# f2 = f0 + (w * f0)/ 2

b = 0.06

g_wb = [1, 1.1897, 1.4346, 2.1199, 1.6010, 2.1699, 1.5640, 1.9444, 0.8778, 1.3554]
g_nb = [1, 1.1681, 1.4039, 2.0562, 1.5170, 1.9029, 0.8618, 1.3554]

g = [1, 1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0000]
Ck, Cmk = rfn.utils.combline_sections_nb(g, f1, f2, er=er, h=0.25)

# print(Ck, Cmk)

def find_Cab_spacing(sp, target_Cab, w):
   Cf_o, Cf_e = utils.coupled_sline_fringing_cap(w, sp, b, er)
   # normalized Cab
   Cab = (Cf_o - Cf_e) / (e0 * er)

   return Cab - target_Cab

# determine spacings using the capacitance between lines Cmk
sk = np.zeros_like(Cmk)
wk = np.ones_like(Ck) * (b / 2)

for m in range(10):
    for i, cmk in enumerate(Cmk):
        w = wk[i+1] if i < len(wk) - 2 else wk[i-1]
        sk[i] = least_squares(find_Cab_spacing, x0=b*0.5, args=(cmk, w), bounds=(0.001, b)).x[0]

    # even mode fringing capacitance for each space between lines, width is arbitrary here
    Cf_e = np.array([
        utils.coupled_sline_fringing_cap(wk[i+1] if i < len(wk) - 2 else wk[i-1], s, b, er)[1] for i, s in enumerate(sk)
    ]) / (e0 * er)

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

print(wk[1:-1], sk[1:-1])

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
y0 = 0.06 * ymax
y1 = ymax - y0


w0 = 0.035
feed_y = 0.37
feed_x = 0.06
y1_tune = 0.065
sk_adjust = 0.000


# x coordinates of the left and right edge of each line
x0_k = np.zeros(K)
x1_k = np.zeros(K)
y0_k = np.zeros(K)
y1_k = np.zeros(K)

x0_k[0] = 0.15
x1_k[0] = x0_k[0] + wk[0]

# wk[1] -= 0.004
# wk[-2] -= 0.004

for i in range(K):
    if i % 2: # if odd
        y0_k[i] = 0
        y1_k[i] = y1
    else: # if even
        y0_k[i] = y0
        y1_k[i] = ymax

for i in range(1, K):
    x0_k[i] = x1_k[i-1] + sk[i-1]
    x1_k[i] = x0_k[i] + wk[i]

# adjust outer lines
y1_k[1] += y1_tune
y1_k[-2] += y1_tune


sbox_w = x1_k[-1] + 0.15
sbox_len = ymax
sbox_h = b
substrate = pv.Cube(center=(sbox_w/2, sbox_len/2, 0), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)
sbox =      pv.Cube(center=(sbox_w/2, sbox_len/2, 0), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)

s = rfn.Solver_3D(sbox)
s.add_dielectric("sub", substrate, er=er, loss_tan=0.003, f0=1.5e9, style=dict(opacity=0.0))

# add resonators. Skip the first and last line as these are impedance transformers and we're using the tap instead
for i in range(1, K-1):
    line = pv.Rectangle([
        (x0_k[i], y0_k[i], 0),
        (x1_k[i], y0_k[i], 0),
        (x1_k[i], y1_k[i], 0),
    ])

    s.add_conductor(f"line_{i}", line, style=dict(color="gold"))

# add feed lines
rfn.elements.Stripline(w=0.035, b=b, er=er).get_properties(f0)

feed_1 = pv.Rectangle([
        (feed_x, feed_y-w0/2, 0),
        (feed_x, feed_y+w0/2, 0),
        (x0_k[1], feed_y+w0/2, 0),
])

feed_2 = pv.Rectangle([
        (sbox_w-feed_x, feed_y-w0/2, 0),
        (sbox_w-feed_x, feed_y+w0/2, 0),
        (x1_k[-2], feed_y+w0/2, 0),
])

s.add_conductor(f"feed_1", feed_1, style=dict(color="gold"))
s.add_conductor(f"feed_2", feed_2, style=dict(color="gold"))


port1_face = pv.Rectangle([
    (feed_x, feed_y-w0/2, 0),
    (feed_x, feed_y+w0/2, 0),
    (feed_x, feed_y+w0/2, -sbox_h/2),
])

port2_face = pv.Rectangle([
    (feed_x, feed_y-w0/2, 0),
    (feed_x, feed_y+w0/2, 0),
    (feed_x, feed_y+w0/2, sbox_h/2),
])


port3_face = pv.Rectangle([
    (sbox_w-feed_x, feed_y-w0/2, 0),
    (sbox_w-feed_x, feed_y+w0/2, 0),
    (sbox_w-feed_x, feed_y+w0/2, -sbox_h/2),
])


port4_face = pv.Rectangle([
    (sbox_w-feed_x, feed_y-w0/2, 0),
    (sbox_w-feed_x, feed_y+w0/2, 0),
    (sbox_w-feed_x, feed_y+w0/2, sbox_h/2),
])

s.add_lumped_port(1, port1_face, r0=100)
s.add_lumped_port(2, port2_face, r0=100)
s.add_lumped_port(3, port3_face, r0=100)
s.add_lumped_port(4, port4_face, r0=100)


# s.init_mesh(d0 = lam0/20, n0 = 2, d_pec = lam0/20, n_min_pec=4, d_sub=lam0/20, n_min_sub=4, blend_pec=True)
s.generate_mesh(d0 = 0.02, d_edge = 0.005, z_bounds=[0.01, 0.01])



self = s

# plotter = s.render(show_probes=False)
# plotter.camera_position = "xy"
# plotter.show()
# print(s.Nx * s.Ny * s.Nz / 1e3, "kcells")


s.add_field_monitor("mon1", "ez", "z", 0, 100)
# s.add_field_monitor("mon1", "ey", "z", sub_h, 5)
# s.add_field_monitor("mon2", "ey", "z", sub_h, 15)
# s.add_field_monitor("mon3", "ex", "z", sub_h, 10)

pulse_n = 100000
# # width of half pulse in time
# t_half = (s.dt * 100)
# # center of the pulse in time
# t0 = (s.dt * 400)

vsrc = 1e-2 * s.gaussian_source(s.dt * 300, t0= s.dt * 200, t_len = s.dt * pulse_n)
# vsrc = 1e-2 * s.gaussian_modulated_source(f0, width=s.dt * 10000, t0=s.dt * 5000, t_len = pulse_n * s.dt)

# t = np.linspace(0, s.dt * pulse_n, pulse_n)
# vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

frequency: np.ndarray = np.arange(0.5e9, 3e9, 2e6)

s.run([1, 2], [vsrc, -vsrc], n_threads=4)
self = s


# p = s.plot_monitor(
#     ["mon1"], zoom=1.5, view="yz", el=30, opacity=[0.9, 1], az=-70,
#     linear=False, cmap="jet", style="surface",  gif_file=dir_ / "combline.gif"
# )
# # p.camera_position = "xy"
# p.show(title="EM Solver")


sdata = s.get_sparameters(frequency, 1, z0=100)
S11 = sdata[:, 0]
S21 = sdata[:, 2]


fig, axes = plt.subplots(2, 2, figsize=(9, 9))

ax = axes[1,0]
rfn.plots.draw_smithchart(ax)
ax.plot(S11.real, S11.imag)

ax = axes[0,1]
ax.plot(frequency / 1e9, conv.db20_lin(S11))
ax.set_xlabel("Frequency [GHz]")
ax.set_xticks(np.arange(0.6, 2.6, 0.2))
ax.set_xlim([0.6, 2.4])
ax.set_ylabel("[dB]")
ax.set_ylim([-40, 2])
ax.grid(True)
ax.legend(["S11", "Ref"])

ax = axes[0,0]
ax.plot(frequency / 1e9, conv.db20_lin(S21))
mplm.line_marker(x = f0/1e9, axes=ax)
ax.set_xticks(np.arange(0.6, 2.6, 0.2))
ax.set_xlim([0.6, 2.4])
ax.set_ylim([-60, 2])
ax.grid(True)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S21", "Ref"])

ax = axes[1,1]
ax.plot(frequency / 1e9, np.unwrap(np.angle(S21, deg=True)))
ax.set_xticks(np.arange(0.6, 2.6, 0.2))
ax.set_xlim([0.6, 2.4])
mplm.line_marker(x = f0/1e9, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[deg]")
ax.legend(["S21", "Ref"])

fig.tight_layout()
plt.show()