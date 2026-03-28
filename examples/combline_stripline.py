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

from scipy import optimize

# pv.set_jupyter_backend("trame")
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

b = 0.06

g = rfn.utils.chebyshev_prototype(5, ripple=0.25)
Ck, Cmk = rfn.utils.combline_sections_nb(g, f1, f2, er=er, h=0.25)

Cmk = (Cmk * 0.85)

def Cab_error(sp, target_Cab, w):
   Cf_o, Cf_e = utils.coupled_sline_fringing_cap(w, sp, b, er)
   # normalized Cab
   Cab = (Cf_o - Cf_e) / (e0 * er)
   # return error between realized interline capacitance and target value
   return Cab - target_Cab

# determine spacings using the capacitance between lines Cmk
sk = np.zeros_like(Cmk)
wk = np.ones_like(Ck) * (b / 2)

for m in range(5):
    for i, cmk in enumerate(Cmk):
        w = wk[i+1] if i < (len(wk) - 1) else wk[i-1]
        sk[i] = optimize.least_squares(Cab_error, x0=b*0.5, args=(cmk, w), bounds=(0.001, b)).x[0]

    # even mode fringing capacitance for each space between lines, width is arbitrary here
    Cf_e = np.array([
        utils.coupled_sline_fringing_cap(wk[i+1] if i < len(wk) - 2 else wk[i-1], s, b, er)[1] for i, s in enumerate(sk)
    ]) / (e0 * er)

    # fringing capacitance on the outer edges (not between the two lines), figure 5.05-10b, for t=0
    Cf = 0.44
    # Ck is the even mode capacitance Ce. Use equation 5.05-25 to determine the per unit length parallel plate 
    # capacitance for each line. Normalized by eps
    Cf_eps_left = np.concatenate([[Cf], Cf_e])
    Cf_eps_right = np.concatenate([Cf_e, [Cf]])
    Cp_e = (Ck / 2) - Cf_eps_left - Cf_eps_right
    # determine width using parallel plate capacitance Cp_e = 2w / b
    wk = (Cp_e / 2) * b


wk = wk[1:-1]
sk = sk[1:-1]
K = len(wk)

# y coordinates of the bottom and top edge of each line
ymax = rfn.const.c0_in / (np.sqrt(f1 * f2) * np.sqrt(er) * 4)

y0 = 0.095
y1 = ymax - y0
via_size = 0.03

w0 = 0.035
feed_y = 0.31
feed_len = 0.12

# sk = np.array(sk) + 0.003
# wk = np.array(wk) + 0.003

# sk[1] += 0.002
# sk[4] += 0.002

# sk[0] += 0.001
# sk[2] += 0.003
# sk[3] += 0.003
# sk[3] += 0.001

# values that work
ref_wk = np.array([0.02266854, 0.02478139, 0.02500937, 0.02477977, 0.02266854])
ref_sk = np.array([0.01674886, 0.01947338, 0.01947338, 0.01674886])

# [0.01966854 0.02178139 0.02200937 0.02177977 0.01966854]
# [0.01574886 0.01947338 0.01947338 0.01574886]

# print("wk", wk - ref_wk)
print("sk", sk - ref_sk)

print(sk)
print(wk)

# x coordinates of the left and right edge of each line
x0_k = np.zeros(K)
x1_k = np.zeros(K)
y0_k = np.zeros(K)
y1_k = np.zeros(K)



for i in range(K):
    if i % 2: # if odd
        y0_k[i] = y0
        y1_k[i] = ymax
    else: # if even
        y0_k[i] = 0
        y1_k[i] = y1

x0_k[0] = 0.1 + feed_len
x1_k[0] = x0_k[0] + wk[0]

for i in range(1, K):
    x0_k[i] = x1_k[i-1] + sk[i-1]
    x1_k[i] = x0_k[i] + wk[i]

y1_k[0] += 0.065
y0_k[1] += 0.01
# y1_k[2] -= 0.02
y0_k[3] += 0.01
y1_k[4] += 0.065

sbox_w = x1_k[-1] + 0.1 + feed_len
sbox_len = ymax + 0.15
sbox_h = b
substrate = pv.Cube(center=(sbox_w/2, sbox_len/2 - (0.15/2), 0), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)
sbox =      pv.Cube(center=(sbox_w/2, sbox_len/2 - (0.15/2), 0), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)

s = rfn.FDTD_Solver(sbox)
s.add_dielectric(substrate, er=er, loss_tan=0.003, f0=np.sqrt(f1 * f2), style=dict(opacity=0.0))

# add resonators. Skip the first and last line as these are impedance transformers and we're using the tap instead
for i in range(K):
    line = pv.Rectangle([
        (x0_k[i], y0_k[i], 0),
        (x1_k[i], y0_k[i], 0),
        (x1_k[i], y1_k[i], 0),
    ])

    s.add_conductor(line, style=dict(color="gold"))

    # add shorting vias, bottom of resonator if odd, top otherwise
    if i % 2:
        via = pv.Box(
            (x0_k[i], x1_k[i], y1_k[i], y1_k[i] + via_size, -sbox_h/2, sbox_h/2)
        )
    else:
        via = pv.Box(
            (x0_k[i], x1_k[i], y0_k[i] - via_size, y0_k[i], -sbox_h/2, sbox_h/2)
        )


    s.add_conductor(via, style=dict(color="gold", opacity=0.6))

feed_1 = pv.Rectangle([
        (x0_k[0] - feed_len, feed_y-w0/2, 0),
        (x0_k[0] - feed_len, feed_y+w0/2, 0),
        (x0_k[0], feed_y+w0/2, 0),
])

feed_2 = pv.Rectangle([
        (x1_k[-1] + feed_len, feed_y-w0/2, 0),
        (x1_k[-1] + feed_len, feed_y+w0/2, 0),
        (x1_k[-1], feed_y+w0/2, 0),
])

s.add_conductor(feed_1, style=dict(color="gold"))
s.add_conductor(feed_2, style=dict(color="gold"))


port1_face = pv.Rectangle([
    (x0_k[0] - feed_len, feed_y-w0/2, -sbox_h/2),
    (x0_k[0] - feed_len, feed_y+w0/2, -sbox_h/2),
    (x0_k[0] - feed_len, feed_y+w0/2, sbox_h/2),
])

port2_face = pv.Rectangle([
    (x1_k[-1] + feed_len, feed_y-w0/2, -sbox_h/2),
    (x1_k[-1] + feed_len, feed_y+w0/2, -sbox_h/2),
    (x1_k[-1] + feed_len, feed_y+w0/2, sbox_h/2),
])

integration_line1 = pv.Line((x0_k[0] - feed_len, feed_y, -sbox_h/2), (x0_k[0] - feed_len, feed_y, 0))
integration_line2 = pv.Line((x1_k[-1] + feed_len, feed_y, -sbox_h/2), (x1_k[-1] + feed_len, feed_y, 0))
s.add_lumped_port(1, port1_face, integration_line=integration_line1)
s.add_lumped_port(2, port2_face, integration_line=integration_line2)

plotter = s.render(show_probes=False, show_mesh=False)
plotter.camera_position = "xy"
plotter.show()

s.generate_mesh(d0 = 0.02, d_edge = 0.005)

for i in range(K):
    if i == 0:
        s.edge_correction(
            (x0_k[i], y0_k[i], 0), 
            (x0_k[i], feed_y-w0/2, 0), 
            integration_line="x-"
        )
        s.edge_correction(
            (x0_k[i], y1_k[i], 0), 
            (x0_k[i], feed_y+w0/2, 0), 
            integration_line="x-"
        )
    else:
        s.edge_correction(
            (x0_k[i], y0_k[i], 0), 
            (x0_k[i], y1_k[i], 0), 
            integration_line="x-"
        )

    if i == (K - 1):
        s.edge_correction(
            (x1_k[i], y0_k[i], 0), 
            (x1_k[i], feed_y-w0/2, 0), 
            integration_line="x+"
        )
        s.edge_correction(
            (x1_k[i], feed_y+w0/2, 0), 
            (x1_k[i], y1_k[i], 0), 
            integration_line="x+"
        )
    else:
        s.edge_correction(
            (x1_k[i], y0_k[i], 0), 
            (x1_k[i], y1_k[i], 0), 
            integration_line="x+"
        )


# plotter = s.render(show_probes=False)
# plotter.camera_position = "xy"
# plotter.show()
# print(s.Nx * s.Ny * s.Nz / 1e3, "kcells")


# p = s.plot_coefficients("ey_z", "a", "z", 0, point_size=15, cmap="brg")
# p.camera_position = "xy"
# p.show()

# s.add_field_monitor("mon1", "e_total", "z", 0, 100)
# s.add_field_monitor("mon1", "ey", "z", sub_h, 5)
# s.add_field_monitor("mon2", "ey", "z", sub_h, 15)
# s.add_field_monitor("mon3", "ex", "z", sub_h, 10)

pulse_n = 50000
# # width of half pulse in time
# t_half = (s.dt * 100)
# # center of the pulse in time
# t0 = (s.dt * 400)

# vsrc = 1e-2 * s.gaussian_source(s.dt * 300, t0= s.dt * 200, t_len = s.dt * pulse_n)
vsrc = 1e-2 * s.gaussian_source(width=s.dt * 1000, t0=s.dt * 1000, t_len = pulse_n * s.dt)

# t = np.linspace(0, s.dt * pulse_n, pulse_n)
# vsrc = 1e-2 * (np.sin(2* np.pi * f0 * (t)) * np.exp(-((t - t0) / t_half)**2)).astype(np.float32)
plt.plot(vsrc)

frequency: np.ndarray = np.arange(0.5e9, 3e9, 2e6)

s.assign_excitation(vsrc, 1)
s.solve(n_threads=4)
self = s


# p = s.plot_monitor(["mon1"], camera_position="xy", opacity=0.8, gif_setup=None)
# p.show(title="EM Solver")
# # # p.camera_position = "xy"
# # p.show(title="EM Solver")


sdata = s.get_sparameters(frequency, 1, downsample=True)
S11 = sdata[:, 0]
S21 = sdata[:, 1]


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9), height_ratios=[1, 2], constrained_layout=True)

rfn.plots.draw_smithchart(ax2)
ax2.plot(S11.real, S11.imag)

ax1.plot(frequency / 1e9, conv.db20_lin(S11))
ax1.plot(frequency / 1e9, conv.db20_lin(S21))
ax1.set_xlabel("Frequency [GHz]")
ax1.set_xticks(np.arange(0.6, 2.6, 0.2))
ax1.set_xlim([0.6, 2.4])
ax1.set_ylabel("[dB]")
ax1.set_ylim([-40, 2])
ax1.grid(True)
ax1.legend(["S11", "S21"])

fig.tight_layout()
plt.show()