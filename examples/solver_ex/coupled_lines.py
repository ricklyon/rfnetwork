import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import rfnetwork as rfn
import mpl_markers as mplm
from scipy.special import ellipk


import sys
# matplotlib.use("qt5agg")

pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)

sys.argv = sys.argv[0:1]

u0 = const.u0
e0 = const.e0
c0 = const.c0


# remove e edge conductive on x sides as well if there is no port attached to it
# allocate block memory in python and assign section to each thread, that way multiple runs can share the same
# memory without reallocating it.

ms_w = 0.04
ms_sp = 0.01

ms1_y = -(ms_w / 2) - (ms_sp / 2)
ms2_y = (ms_w / 2) + (ms_sp / 2)

sbox_h = 0.25
sbox_w = 0.4
sbox_len = 0.5

sub_h = 0.02
ms_x = (-sbox_len/2 + 0.1, sbox_len/2)

er=3.66
f0 = 10e9

w = ms_w
s = ms_sp
h = sub_h

Zo, Ze = rfn.utils.coupled_ustrip_impedance(w, s, h, er)

print(f"Zo={Zo:.1f}, Ze={Ze:.1f}")


line = rfn.elements.MSLine(h=sub_h, er=er, w=ms_w)
z_ref = line.get_properties(10e9).sel(value="z0").item()

def ustrip_fringing_cap(w, h, er, f0):
    """
    Fringing capacitance from edge of uncoupled microstrip line, per unit length [m].
    This is a weak function of the width of the line.
    """
    # characteristic impedance and effective epsilon
    zc, er_eff, _ = rfn.elements.MSLine(h, er, w).get_properties(f0).squeeze()

    # parallel plate capacitance
    Cp = er * e0 * (w / h)
    # uncoupled fringe capacitance
    Cf = ((np.sqrt(er_eff) / (c0 * zc)) - Cp) / 2

    # fringing capacitance seems to be a bit high, reduced empirically 
    return Cf * 0.85

def coupled_ustrip_fringing_cap(w: float, s: float, h: float, er: float, f0):
    """
    Odd and even mode fringing capacitance for edge coupled microstrip line, per unit length [m].
    See microstrip_coupled_capacitance.pdf in docs/solver

    Assumes that thickness is zero.
    """
    # uncoupled fringing capacitance
    Cf = ustrip_fringing_cap(w, h, er, f0)

    # fringing capacitance in even mode
    A = np.exp(-0.1 * np.exp(2.33 - 2.53 * (w / h)))
    Cf_e = Cf / (1 + A * (h / s) * np.tanh(8 * s / h))

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

def coupled_ustrip_cap(w: float, s: float, h: float, er: float, f0: float):
    """ Total odd and even mode capacitance per unit length """
    Cf = ustrip_fringing_cap(w, s, er, f0)

    Cfo, Cfe = coupled_ustrip_fringing_cap(w, s, h, er, f0)

    Cp = er * e0 * w / h

    # total odd and even mode capacitances
    Co = Cp + Cf + Cfo
    Ce = Cp + Cf + Cfe

    return Co, Ce

def coupled_ustrip_impedance(w: float, s: float, h: float, er: float, f0: float):
    """
    Odd and even mode total capacitance for edge coupled microstrip line, per unit length [m].
    See microstrip_coupled_capacitance.pdf in docs/solver
    """

    # total even and odd mode coupled capacitance
    Co, Ce = coupled_ustrip_cap(w, s, h, er, f0)
    # coupled capacitance in air
    Co_a, Ce_a = coupled_ustrip_cap(w, s, h, 1.001, f0)

    Zo = 1 / (const.c0 * np.sqrt(Co * Co_a))
    Ze = 1 / (const.c0 * np.sqrt(Ce * Ce_a))

    return Zo, Ze

w = ms_w
s = ms_sp
h = sub_h

# Zo, Ze = coupled_ustrip_impedance(w, s, h, er, f0)
# print(f"Zo={Zo:.1f}, Ze={Ze:.1f}")


substrate = pv.Cube(center=(0, 0, sub_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sub_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

ms1_trace = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[1], ms1_y + ms_w/2, sub_h)
])

ms2_trace = pv.Rectangle([
    (ms_x[0], ms2_y - ms_w/2, sub_h),
    (ms_x[0], ms2_y + ms_w/2, sub_h),
    (ms_x[1], ms2_y + ms_w/2, sub_h)
])

port1_face = pv.Rectangle([
    (ms_x[0], ms1_y - ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, sub_h),
    (ms_x[0], ms1_y + ms_w/2, 0),
])

port2_face = pv.Rectangle([
    (ms_x[0], ms2_y - ms_w/2, sub_h),
    (ms_x[0], ms2_y + ms_w/2, sub_h),
    (ms_x[0], ms2_y + ms_w/2, 0),
])


current_face = pv.Rectangle([
    (0, ms1_y - ms_w/2, sub_h + 0.001),
    (0, ms1_y + ms_w/2, sub_h + 0.001),
    (0, ms1_y + ms_w/2, sub_h - 0.001),
])


voltage_line = pv.Line(
    [0, ms1_y, 0], [0, ms1_y, sub_h]
)

s = rfn.FDTD_Solver(sbox)
s.add_dielectric(substrate, er=er, style=dict(opacity=0.0))
s.add_conductor(ms1_trace, style=dict(color="gold"))
s.add_conductor(ms2_trace, style=dict(color="gold"))
s.add_lumped_port(1, port1_face, "z-")
s.add_lumped_port(2, port2_face, "z-")


self = s
s.assign_PML_boundaries("x+", n_pml=10)

s.generate_mesh(d0 = 0.02, d_edge = 0.0025)


s.add_field_monitor("mon1", "ez", "z", sub_h - 0.005, 10)
s.add_field_monitor("mon2", "ey", "z", sub_h, 10)
s.add_field_monitor("mon3", "ex", "z", sub_h, 10)

s.add_current_probe("c1", current_face)
s.add_voltage_probe("v1", voltage_line)


p1 = (ms_x[0], ms1_y + ms_w/2, sub_h)
p2 = (ms_x[1], ms1_y + ms_w/2, sub_h)
integration_line = "y+"

s.edge_correction(p1, p2, "y+")

p1 = (ms_x[0], ms1_y - ms_w/2, sub_h)
p2 = (ms_x[1], ms1_y - ms_w/2, sub_h)

s.edge_correction(p1, p2, "y-")

p1 = (ms_x[0], ms2_y + ms_w/2, sub_h)
p2 = (ms_x[1], ms2_y + ms_w/2, sub_h)
integration_line = "y+"

s.edge_correction(p1, p2, "y+")

p1 = (ms_x[0], ms2_y - ms_w/2, sub_h)
p2 = (ms_x[1], ms2_y - ms_w/2, sub_h)

s.edge_correction(p1, p2, "y-")


plotter = s.render(show_probes=True)
# plotter.camera_position = "yz"
plotter.show()


# # print(s.Nx, s.Ny, s.Nz)

# # p.camera.zoom(1.8)

# Db_0 = s.dt / u0
# Cb_0 = s.dt / e0 
p = s.plot_coefficients("hz_y1", "b", "z", sub_h, point_size=15, cmap="brg")
p.camera_position = "xy"
p.show()

f0 = 10e9
vsrc = 1e-2 * self.gaussian_source(width=50e-12, t0=80e-12, t_len=200e-12)
plt.plot(vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.assign_excitation(vsrc, [1, 2])
# run even mode
s.solve(n_threads=4)

sdata = s.get_sparameters(frequency)
S11_even = sdata[:, 0]

line_i = self.vi_probe_values("c1")
line_v = self.vi_probe_values("v1")

IP = utils.dtft(-self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt)
ZP_even = VP / IP

# run odd mode
s.assign_excitation(vsrc, 1)
s.assign_excitation(-vsrc, 2)
s.solve(n_threads=4)

sdata = s.get_sparameters(frequency)
S11_odd = sdata[:, 0]

p = s.plot_monitor(["mon1"])
p.show(title="EM Solver")


# fig, ax = plt.subplots()
# rfn.plots.draw_smithchart(ax)
# plt.plot(S11.real, S11.imag)

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v = self.vi_probe_values("v1")

IP = utils.dtft(-self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(self.vi_probe_values("v1"), frequency, 1 / s.dt)
ZP_odd = VP / IP

fig, ax = plt.subplots()
ax.plot(frequency / 1e9, ZP_even.real)
ax.plot(frequency / 1e9, ZP_odd.real)
ax.plot(frequency / 1e9, conv.z_gamma(S11_even))
ax.plot(frequency / 1e9, conv.z_gamma(S11_odd))
plt.ylim([0, 110])
# plt.axhline(y=z_ref, linestyle=":", color="k")
plt.axhline(y=Ze, linestyle=":", color="k")
plt.axhline(y=Zo, linestyle=":", color="k")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
ax.legend(["Even Mode", "Odd Mode", "Even Mode", "Odd Mode"])
mplm.line_marker(x = 10, axes=ax)

# S11_z = conv.gamma_z(ZP)

# fig, ax = plt.subplots()
# rfn.plots.draw_smithchart(ax)
# plt.plot(S11.real, S11.imag)
# plt.plot(S11_z.real, S11_z.imag)

# fig, ax = plt.subplots()
# plt.plot(frequency, conv.db20_lin(S11))
plt.show()