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

from scipy.optimize import fsolve
from scipy.special import ellipk

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

h = 0.03

g_wb = [1, 1.1897, 1.4346, 2.1199, 1.6010, 2.1699, 1.5640, 1.9444, 0.8778, 1.3554]
g_nb = [1, 1.1681, 1.4039, 2.0562, 1.5170, 1.9029, 0.8618, 1.3554]

g = [1, 0.5176, 1.4142, 1.9318, 1.9318, 1.4142, 0.5176, 1.0000]
Ck, Cmk = rfn.utils.combline_sections_wb(g, f1, f2, er=er, h=0.2)

# print(Ck, Cmk)

def ustrip_fringing_cap(w, h, er):
    """
    Fringing capacitance from edge of uncoupled microstrip line, per unit length [m].
    This is a weak function of the width of the line.
    """
    # characteristic impedance and effective epsilon
    zc, er_eff, _ = rfn.elements.MSLine(h, er, w).get_properties(f0).squeeze()

    # parallel plate capacitance
    Cp = er * e0 * (w / h)
    # uncoupled fringe capacitance
    Cf = (np.sqrt(er_eff) / (c0 * zc)) - Cp

    return Cf

def coupled_ustrip_fringing_cap(w: float, s: float, h: float, er: float):
    """
    Odd and even mode fringing capacitance for edge coupled microstrip line, per unit length [m].
    See microstrip_coupled_capacitance.pdf in docs/solver

    Assumes that thickness is zero.
    """
    # uncoupled fringing capacitance
    Cf = ustrip_fringing_cap(w, h, er)

    # fringing capacitance in even mode
    A = np.exp(-0.1 * np.exp(2.33 - 2.53 * (w / h)))
    Cf_e = Cf / (1 + A * (h / s) * np.arctan(8 * s / h))

    # odd mode fringing capacitance through the dielectric
    coth = lambda x : np.cosh(x) / np.sinh(x)
    C_gd = (e0 * er / np.pi) * np.log(coth((np.pi / 4) * (s / h))) + 0.65 * Cf * (
        ((0.02 * np.sqrt(er)) / (s / h)) + 1 - (1 / (er **2 ))
    )
    # odd mode fringing capacitance through the air
    k = (s / h) / ((s / h) + 2 * (w / h))
    kp = np.sqrt(1 - (k**2))
    C_ga = e0 * ellipk(kp) / ellipk(k)

    # return unnormalized capacitance in farads, odd, even
    return (C_gd + C_ga), Cf_e

def find_Cab_spacing(sp, target_Cab):
    w = h
    Cf_o, Cf_e = coupled_ustrip_fringing_cap(w, sp, h, er)
    # normalized interspace capacitance
    # Derive this from Co = Ca + 2 Cab, using the equations for Co and Ce (Ce=Ca) in the reference linked in 
    # coupled_usline_fringing_cap
    Cab = (Cf_o - Cf_e) / (2 * e0 * er)

    return Cab - target_Cab

# determine spacings using the capacitance between lines Cmk
sk = np.zeros_like(Cmk)
for i, cmk in enumerate(Cmk):
    sk[i] = least_squares(find_Cab_spacing, x0=h*0.5, args=(cmk,), bounds=(0.001, h*3)).x[0]


# even mode fringing capacitance for each space between lines, width is arbitrary here
# capacitance is normalized by epsilon
Cf_e = np.array([coupled_ustrip_fringing_cap(h, s, h, er)[1] for s in sk]) / (e0 * er)
# fringing capacitance on the outer edges (not between the two lines), figure 5.05-10b, for t=0
Cf = ustrip_fringing_cap(h, h, er) / (e0 * er)
# Ck is the even mode capacitance Ce. Ce = Cp + Cf + Cf_e, Ce=Ca=Ck
Cf_eps_left = np.concatenate([[Cf], Cf_e])
Cf_eps_right = np.concatenate([Cf_e, [Cf]])
Cp = Ck - Cf_eps_left - Cf_eps_right
# determine width using parallel plate capacitance Cp = er * e0 * (w / h)
wk = Cp * h


# use multiple iterations where actual widths are fed back into the algorithm
print(wk, sk)

# manual adjustmment
sk[0] = 0.012
sk[-1] = 0.012

# design taken from table 10.07-2 in Matthaei
K = len(wk)
#k       0      1      2      3      4      5      6      7
# w_k =   [0.126, 0.121, 0.126, 0.127, 0.127, 0.126, 0.121, 0.126]
# s_k =   [0.092, 0.136, 0.143, 0.146, 0.143, 0.136, 0.087]

# y coordinates of the bottom and top edge of each line
zc, er_eff, _ = rfn.elements.MSLine(h, er, w=0.02).get_properties(f0).squeeze()
ymax = rfn.const.c0_in / (f0 * np.sqrt(er_eff) * 4)
y0 = 0.05 * ymax
y1 = ymax - y0

# x coordinates of the left and right edge of each line
x0_k = np.zeros(K)
x1_k = np.zeros(K)
y0_k = np.zeros(K)
y1_k = np.zeros(K)

x0_k[0] = 0.2
x1_k[0] = x0_k[0] + wk[0]

for i in range(K):
    if i % 2: # if odd
        y0_k[i] = y0
        y1_k[i] = ymax
    else: # if even
        y0_k[i] = 0
        y1_k[i] = y1

# y0_k[0] = 0.025 * ymax
# y1_k[-1] = ymax - (0.025 * ymax)

for i in range(1, K):
    x0_k[i] = x1_k[i-1] + sk[i-1]
    x1_k[i] = x0_k[i] + wk[i]

sbox_w = x1_k[-1] + 0.2
sbox_len = ymax
sbox_h = h * 6
substrate = pv.Cube(center=(sbox_w/2, sbox_len/2, -h/2), x_length=sbox_w, y_length=sbox_len, z_length=h)
sbox =      pv.Cube(center=(sbox_w/2, sbox_len/2, (sbox_h - h)/2), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h + h)

s = rfn.Solver_PCB(sbox, nports=2)
s.add_substrate("sub", substrate, er=er, opacity=0.0, loss_tan=0.0035, f0=1.5e9)


for i in range(K):
    line = pv.Rectangle([
        (x0_k[i], y0_k[i], 0),
        (x1_k[i], y0_k[i], 0),
        (x1_k[i], y1_k[i], 0),
    ])

    s.add_pec_face(f"line_{i}", line, color="gold")

# feed traces
w0 = rfn.elements.MSLine(h, er, z0 = 50).state["w"]
feed_y = 0.4
port_offset = 0.07

port1_feed = pv.Rectangle([
    (port_offset, feed_y, 0),
    (x0_k[0], feed_y, 0),
    (x0_k[0], feed_y + w0, 0),
])

port1_feed = pv.Rectangle([
    (port_offset, feed_y, 0),
    (x0_k[0], feed_y, 0),
    (x0_k[0], feed_y + w0, 0),
])

port2_feed = pv.Rectangle([
    (sbox_w- port_offset, ymax - feed_y, 0),
    (x1_k[-1], ymax - feed_y, 0),
    (x1_k[-1], ymax - feed_y - w0, 0),
])

s.add_pec_face(f"port1_feed", port1_feed, color="gold")
s.add_pec_face(f"port2_feed", port2_feed, color="gold")


# ports
port1_face = pv.Rectangle([
    (port_offset, feed_y, -h),
    (port_offset, feed_y + w0, -h),
    (port_offset, feed_y + w0, 0),
])

port2_face = pv.Rectangle([
    (sbox_w - port_offset, ymax - feed_y, -h),
    (sbox_w - port_offset, ymax - feed_y - w0, -h),
    (sbox_w - port_offset, ymax - feed_y, 0),
])

s.add_lumped_port(1, port1_face)
s.add_lumped_port(2, port2_face)


# s.init_mesh(d0 = lam0/20, n0 = 2, d_pec = lam0/20, n_min_pec=4, d_sub=lam0/20, n_min_sub=4, blend_pec=True)
s.init_mesh_edge_method(d0 = 0.07, d_edge = 0.005)
s.init_coefficients()

# s.init_mesh_edge_method(d0 = 0.1, d_edge=0.01)
# s.init_coefficients()

self = s

plotter = s.render()
plotter.camera_position = "xy"
plotter.show()
print(s.Nx * s.Ny * s.Nz / 1e3, "kcells")


s.init_ports(r0=50)
s.init_pec()

s.add_field_monitor("mon1", "ez", "z", 0, 30)
# s.add_field_monitor("mon1", "ey", "z", sub_h, 5)
# s.add_field_monitor("mon2", "ey", "z", sub_h, 15)
# s.add_field_monitor("mon3", "ex", "z", sub_h, 10)

pulse_n = 50000
# # width of half pulse in time
# t_half = (s.dt * 100)
# # center of the pulse in time
# t0 = (s.dt * 400)

vsrc = 1e-2 * s.gaussian_source(s.dt * 300, s.dt * pulse_n)

# t = np.linspace(0, s.dt * pulse_n, pulse_n)
# vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

frequency: np.ndarray = np.arange(0.4e9, 6e9, 2e6)

s.run([1], [vsrc], n_threads=4)
self = s


p = s.plot_monitor(
    ["mon1"], zoom=1.1, view="xy", el=0, opacity=[0.9, 1], 
    linear=False, cmap="jet", style="surface",
)
# p.camera_position = "xy"
p.show(title="EM Solver")


sdata = s.get_sparameters(frequency, 1, z0=50)
S11 = sdata[:, 0]
S21 = sdata[:, 1]


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