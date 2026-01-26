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

er = 1

f1 = 1e9
f2 = 2e9

f0 = (f1 + f2) / 2
w = (f2 - f1) / f0

wp = 1

lam0 = rfn.const.c0_in / f0

n = 8
theta_1 = (np.pi / 2) * (1 - (w / 2))
Y_a = (1 / 50)

# page 627 example
g = [1, 1.1897, 1.4346, 2.1199, 1.6010, 2.1699, 1.5640, 1.9444, 0.8778, 1.3554]

# Table 10.07-1, pg 628
Jk2_Y = [g[2] / ( g[0] * np.sqrt(g[k] * g[k+1]) ) for k in range(2, n-2)]
Jn_Y = (1 / g[0]) * np.sqrt((g[2] * g[0]) / (g[n-2] * g[n+1]))
Jk_Y = [0, 0] + Jk2_Y + [Jn_Y]

Nk = [0, 0] + [np.sqrt((Jk_Y[k])**2 + ((wp * g[2] * np.tan(theta_1)) / (2 * g[0]))**2) for k in range(2, n-1)]

Zn_Za = [(wp * g[k] * g[k+1] * np.tan(theta_1)) for k in range(n+1)]

Y2_Ya = ((wp * g[2]) / (2 * g[0])) * np.tan(theta_1) + Nk[2] - (Jk_Y[2])
Yk3_Ya = [Nk[k-1] + Nk[k] - Jk_Y[k-1] - Jk_Y[k] for k in range(3, n-1)]
Yn_Ya = ((wp * ( 2 * g[0] * g[n-1] - g[2] * g[n+1]) * np.tan(theta_1)) / (2 * g[0] * g[n+1])) + Nk[n-2] - Jk_Y[n-2]
Yk_Ya = [0, 0] + [Y2_Ya] + Yk3_Ya + [Yn_Ya]

h = 0.18

# self capacitance, normalized by epsilon
C1 = (eta0 / np.sqrt(er)) * Y_a * (1 - np.sqrt(h)) / (Zn_Za[0])
C2 = (eta0 / np.sqrt(er)) * (Y_a * h) * (Yk_Ya[2]) - np.sqrt(h) * (C1)

Ck3 = [(eta0 / np.sqrt(er)) * (Y_a * h) * (Yk_Ya[k]) for k in range(3, n-1)]

CN = (eta0 / np.sqrt(er)) * Y_a * (1 - np.sqrt(h)) / (Zn_Za[-1])
CN_1 = (eta0 / np.sqrt(er)) * (Y_a * h) * (Yk_Ya[n-1]) - np.sqrt(h) * (CN)

Ck = np.array([C1] + [C2] + Ck3 + [CN_1] + [CN])

# mutual capacitance, normalized by epsilon
Cm12 = (eta0 / np.sqrt(er)) * Y_a * (np.sqrt(h) / (Zn_Za[0]))
Cmk2 = [(eta0 / np.sqrt(er)) * (Y_a * h) * (Jk_Y[k]) for k in range(2, n-1)]
CmN = (eta0 / np.sqrt(er)) * Y_a * (np.sqrt(h) / (Zn_Za[-1]))

Cmk = np.array([Cm12] + Cmk2 + [CmN])

def fringing_capacitance_figure(b, w, er):
    """
    Zero thickness curve in Figure 5.05-9. 
    Results should be nearly independent of b and w as long as w is large enough to be lower than 200 ohms or so.
    """

    sp_sweep = np.arange(0.01, 1.5, 0.001) * b
    Cf_o, Cf_e = np.zeros((2, len(sp_sweep)))

    for i, s in enumerate(sp_sweep):
        Cf_o_s, Cf_e_s = utils.coupled_sline_fringing_cap(w, s, b, er)
        # normalize by epsilon, and convert to per unit inch from meters
        Cf_o[i] = Cf_o_s / (e0 * er)
        Cf_e[i] = Cf_e_s / (e0 * er)

    # Cab or del_C is the difference between the odd and fringing capacitance
    Cab = Cf_o - Cf_e

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sp_sweep / b, Cf_o, linewidth=2)
    ax.plot(sp_sweep / b, Cf_e, linewidth=2)
    ax.plot(sp_sweep / b, Cab, linewidth=2)
    ax.set_yscale('log')
    ax.set_ylim([0.01, 10])
    ax.set_xlim([0, 1.5])
    ax.set_xticks(np.arange(0, 1.6, 0.1))
    ax.set_xlabel("s/b")
    ax.set_ylabel(r"C/$\varepsilon$")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10])
    ax.grid(True)
    ax.legend(["$C_{f,o}$", "$C_{f,e}$", "$\Delta C_{f}$"], edgecolor='k', shadow=True, fontsize=12)
    ax.set_title("Normalized Fringing Capacitance for Edge-Coupled Stripline", fontsize=11)

    return fig, ax



b = 0.625
# w = b * 0.5
# er = 1

def find_Cab_spacing(sp, target_Cab):
   w = b * 0.5
   Cf_o, Cf_e = utils.coupled_sline_fringing_cap(w, sp, b, er)
   # normalized Cab
   Cab = (Cf_o - Cf_e) / (e0 * er)

   return Cab - target_Cab

# determine spacings using the capacitance between lines Cmk
sk = np.zeros_like(Cmk)
for i, cmk in enumerate(Cmk):
    sk[i] = least_squares(find_Cab_spacing, x0=b*0.2, args=(cmk,), bounds=(0.001, b)).x[0]

# even mode fringing capacitance for each space between lines, width is arbitrary here
Cf_e = np.array([utils.coupled_sline_fringing_cap(0.5*b, s, b, er)[1] for s in sk]) / (e0 * er)
# fringing capacitance on the outer edges (not between the two lines), figure 5.05-10b, for t=0
Cf = 0.44
# Ck is the even mode capacitance Ce. Use equation 5.05-25 to determine the per unit length parallel plate capacitance 
# for each line. Normalized by eps
Cf_eps_left = np.concatenate([[Cf], Cf_e])
Cf_eps_right = np.concatenate([Cf_e, [Cf]])
Cp_e = (Ck / 2) - Cf_eps_left - Cf_eps_right
# determine width using parallel plate capacitance Cp_e = 2w / b
wk = (Cp_e / 2) * b

# fringing_capacitance_figure(b=b, w=0.3, er=er)


# design taken from table 10.07-2 in Matthaei
K = 8
#k       0      1      2      3      4      5      6      7
# w_k =   [0.126, 0.121, 0.126, 0.127, 0.127, 0.126, 0.121, 0.126]
# s_k =   [0.092, 0.136, 0.143, 0.146, 0.143, 0.136, 0.087]

# y coordinates of the bottom and top edge of each line
ymax = 1.968
y0 = 0.15
y1 = ymax - y0
#k       0      1       2     3       4      5      6      7
y0_k =  [0.1,   y0,     0,    y0,     0,    y0,     0,     y0    ]
y1_k =  [y1,    ymax,   y1,   ymax,   y1,   ymax,   y1,    ymax - 0.1  ]

# x coordinates of the left and right edge of each line
x0_k = np.zeros(K)
x1_k = np.zeros(K)
x0_k[0] = 0.750
x1_k[0] = x0_k[0] + wk[0]


y1 = np.zeros(K)

for i in range(1, K):
    x0_k[i] = x1_k[i-1] + sk[i-1]
    x1_k[i] = x0_k[i] + wk[i]

sbox_w = x1_k[-1] + 0.750
sbox_len = ymax
sbox_h = 0.625
substrate = pv.Cube(center=(sbox_w/2, sbox_len/2, 0), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)
sbox =      pv.Cube(center=(sbox_w/2, sbox_len/2, 0), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)

s = rfn.Solver_PCB(sbox, nports=4)
s.add_substrate("sub", substrate, er=er, opacity=0.0)


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


port3_face = pv.Rectangle([
    (x0_k[-1], y1_k[-1], 0),
    (x1_k[-1], y1_k[-1], 0),
    (x1_k[-1], y1_k[-1], -sbox_h/2),
])


port4_face = pv.Rectangle([
    (x0_k[-1], y1_k[-1], 0),
    (x1_k[-1], y1_k[-1], 0),
    (x1_k[-1], y1_k[-1], sbox_h/2),
])

s.add_lumped_port(1, port1_face)
s.add_lumped_port(2, port2_face)
s.add_lumped_port(3, port3_face)
s.add_lumped_port(4, port4_face)


# s.init_mesh(d0 = lam0/20, n0 = 2, d_pec = lam0/20, n_min_pec=4, d_sub=lam0/20, n_min_sub=4, blend_pec=True)
s.init_mesh_edge_method(d0 = lam0/50, d_edge = 0.01)
s.init_coefficients()

# s.init_mesh_edge_method(d0 = 0.1, d_edge=0.01)
# s.init_coefficients()



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

pulse_n = 8000
# # width of half pulse in time
# t_half = (s.dt * 100)
# # center of the pulse in time
# t0 = (s.dt * 400)

vsrc = 1e-2 * s.gaussian_source(s.dt * 300, s.dt * pulse_n)

# t = np.linspace(0, s.dt * pulse_n, pulse_n)
# vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
# plt.plot(vsrc)

frequency: np.ndarray = np.arange(0.5e9, 3e9, 2e6)

s.run([1, 2], [vsrc, -vsrc], n_threads=4)
self = s


p = s.plot_monitor(
    ["mon1"], zoom=1.1, view="xy", el=0, opacity=[0.5, 1], 
    linear=False, cmap="jet", style="surface", #gif_file=dir_ / "combline.gif"
)
# p.camera_position = "xy"
p.show(title="EM Solver")


sdata = s.get_sparameters(frequency, 1, z0=100)
S11 = sdata[:, 0]
S21 = sdata[:, 2]


fig, axes = plt.subplots(2, 2, figsize=(9, 9))

ax = axes[0,0]
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

ax = axes[1,0]
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