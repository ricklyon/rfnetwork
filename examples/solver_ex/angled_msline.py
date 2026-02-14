import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import time

import rfnetwork as rfn
import mpl_markers as mplm
import sys
import itertools

from scipy.spatial.transform import Rotation

pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)

sys.argv = sys.argv[0:1]

u0 = const.u0
e0 = const.e0
c0 = const.c0

ms_w = 0.04
ms_len0 = 0.15

# angled section length
ms_ang = 45
ms_len1 = 0.5
ms_len1_x = ms_len1 * np.cos(np.deg2rad(ms_ang))
ms_len1_y = ms_len1 * np.sin(np.deg2rad(ms_ang))

sbox_h = 0.35
sbox_w = ms_len1_x + 0.5
sbox_len = ms_len1_y + 0.3

sub_h = 0.02
ms_x = ((-ms_len1_x / 2), (ms_len1_x / 2))
ms_y = (-ms_len1_y / 2, ms_len1_y / 2)


substrate = pv.Cube(center=(0, 0, sub_h/2), x_length=sbox_w, y_length=sbox_len, z_length=sub_h)

sbox = pv.Cube(center=(0, 0, sbox_h/2), x_length=sbox_w, y_length=sbox_len, z_length=sbox_h)

ms0_trace = pv.Rectangle([
    (ms_x[0], ms_y[0] - ms_w/2, sub_h),
    (ms_x[0], ms_y[0] + ms_w/2, sub_h),
    (ms_x[0] - ms_len0, ms_y[0] + ms_w/2, sub_h)
])

ms1_trace = pv.Rectangle([
    (ms_x[1], ms_y[1] - ms_w/2, sub_h),
    (ms_x[1], ms_y[1] + ms_w/2, sub_h),
    (ms_x[1] + ms_len0, ms_y[1] + ms_w/2, sub_h)
])

ms_ang_trace = pv.Rectangle([
    (0, - ms_w/2, sub_h),
    (0, ms_w/2, sub_h),
    (ms_len1, ms_w/2, sub_h)
])

# rotate trace
rot = Rotation.from_euler("z", ms_ang, degrees=True).as_matrix()
ms_ang_trace.points = np.einsum("ij,kj->ki", rot, ms_ang_trace.points)
# translate to position
ms_ang_trace.points += (ms_x[0], ms_y[0], 0)

# corners
corner0 = pv.Triangle([ms_ang_trace.points[0], (ms_x[0], ms_y[0], sub_h), (ms_x[0], ms_y[0] - ms_w/2, sub_h)])

corner1 = pv.Triangle([ms_ang_trace.points[2], (ms_x[1], ms_y[1], sub_h), (ms_x[1], ms_y[1] + ms_w/2, sub_h)])


port1_face = pv.Rectangle([
    (ms_x[0] - ms_len0, ms_y[0] - ms_w/2, sub_h),
    (ms_x[0] - ms_len0, ms_y[0] + ms_w/2, sub_h),
    (ms_x[0] - ms_len0, ms_y[0] + ms_w/2, 0),
])

port2_face = pv.Rectangle([
    (ms_x[1] + ms_len0, ms_y[1] - ms_w/2, sub_h),
    (ms_x[1] + ms_len0, ms_y[1] + ms_w/2, sub_h),
    (ms_x[1] + ms_len0, ms_y[1] + ms_w/2, 0),
])


s = rfn.Solver_PCB(sbox)
s.add_dielectric("sub", substrate, er=3.66, style=dict(opacity=0.0))
s.add_conductor("ms0", ms0_trace, style=dict(color="gold"))
s.add_conductor("ms1", ms1_trace, style=dict(color="gold"))
s.add_conductor("ms_ang", ms_ang_trace, style=dict(color="gold"))
s.add_conductor("corner0", corner0, style=dict(color="gold"))
s.add_conductor("corner1", corner1, style=dict(color="gold"))
s.add_lumped_port(1, port1_face)
s.add_lumped_port(2, port2_face)

# s.assign_PML_boundaries("z+", "y-", "y+", n_pml=5)

self = s

# angled edge mesh generation
d0 = 0.02
d_edge = 0.005


# having three cells in the PEC instead of 4 causes the edge correction to fail
# s.init_mesh(d0 = 0.02, n0 = 3, d_pec = 0.01, n_min_pec=4, d_sub=0.01, n_min_sub=4, blend_pec=False)
s.generate_mesh(d0 = 0.02, d_edge=0.005, z_bounds = [0.005, 0.02])



plotter = s.render()
plotter.camera_position = "xy"
plotter.show()
print(s.Nx * s.Ny * s.Nz / 1e3)


s.add_field_monitor("mon1", "ez", "z", sub_h, 5)
# s.add_field_monitor("mon1", "ez", "z", sub_h - 0.005, 15)
s.add_field_monitor("mon2", "ez", "y", 0, 5)
s.add_field_monitor("mon3", "ex", "z", sub_h, 10)



if not isinstance(d0, list):
    d0 = [d0] * 3

edges = [np.array([], dtype=np.float32) for i in range(3)]

objects = [self.bounding_box] + [cond["obj"] for cond in self.conductor.values()] + [sub["obj"] for sub in self.dielectric.values()]
dtype_ = np.float32

obj = ms_ang_trace

obj_edges = []
obj_edge_points = []
# build list of edge coordinates along each axis
for obj in objects:

    # decremented counter, starts at the number of points in the face, reaches zero after the last point
    # in the face and a new face begins.
    face_n_count = 0
    # first point in a face
    anchor = None
    # iterate through the list of faces points
    for i, p in enumerate(obj.faces):
        # start a new face
        if face_n_count == 0:
            face_n_count = p
            anchor =  obj.faces[i+1]
            continue
        # if on the last point in the face, connect back to the first point in the face
        elif face_n_count == 1:
            # obj_edges.append((p, anchor))
            obj_edges.append((obj.points[p], obj.points[anchor]))
        # connect two points in the face
        else:
            obj_edges.append((obj.points[p], obj.points[obj.faces[i+1]]))

        face_n_count -= 1

obj_edges = np.around(obj_edges, decimals=self._places).astype(np.float32)
# object vertices
obj_points = [np.unique(obj_edges[..., i].flatten()) for i in range(3)]

# break angled edges up into sections
soft_points = [np.array([]), np.array([]), np.array([])]

edge_len = np.diff(obj_edges, axis=1).squeeze()
# compute the area of a box bound by this edge and the cardinal axis. If area is above a certain threshold
# related to d_edge, create soft points along the axis to keep the area below the threshold. 
edge_area = np.prod(edge_len[:, :2], axis=-1)

edge = obj_edges[0]
for i, edge in enumerate(obj_edges):
    if edge_area[i] > d_edge ** 2:
        # break angled edges along x and y into sub-cells separated by d_edge
        nx_ny = np.around(np.abs(edge_len[i][:2]) / d_edge).astype(int)

        for axis in range(2):
            soft_points[axis] = np.append(soft_points[axis], np.linspace(edge[0, axis], edge[1, axis], nx_ny[axis]))

# add edge cells on either side of object points
for axis in range(3):
    for p in obj_points[axis]:
        soft_points[axis] = np.append(soft_points[axis], [p - d_edge, p + d_edge])

# clip outside of the sbox limits and remove repeated values
soft_points = [np.clip(soft_points[axis], self.sbox_min[axis], self.sbox_max[axis]) for axis in range(3)]
# clean up soft points, round to nearest multiple of d_edge
soft_points = [utils.round_to_multiple(soft_points[axis], d_edge / 4) for axis in range(3)]

# combine object points with soft points
all_points = [np.concatenate([obj_points[axis], soft_points[axis]]) for axis in range(3)]
# round to tolerance
all_points = [np.around(all_points[axis], decimals=self._places) for axis in range(3)]
# remove repeated values and sort
all_points = [np.sort(np.unique(all_points[axis])) for axis in range(3)]

d_bounds = [[d0[i], d0[i]] for i in range(3)]
# build a list of cell widths along each axis
mesh_cells_d = [[], [], []]
for axis in range(3):

    # list of distances between each edge
    cells_d = np.diff(all_points[axis])

    # cells are broken up into smaller sub-cells to create small edge cells, and to shorten the span
    # that is graded with increasingly larger cells.
    subcells_d = []

    for i, d in enumerate(cells_d):

        # split cells larger than d0*5 into multiple cells, this prevents grading over large spans
        split_threshold = d0[axis] * 3
        if int(d / split_threshold) >= 1:
            n_split = int(d / split_threshold)
            # split cell width into equal parts
            subcells_d_i = [d / n_split] * n_split
        else:
            n_split = 1
            subcells_d_i = [d]

        subcells_d += subcells_d_i

    # cells indices arranged from largest width to smallest
    # blend large cells first so they transition gradually into smaller ones.
    # The larger cells have more room to work in and
    # it's easier to blend. The smaller cells have less work to do because the
    # larger cells have already stepped down to meet their widths.
    # Leave PEC cells for last so they don't blend up to the (typically) larger d0 cells
    d_order = np.flip(np.argsort(subcells_d))
    
    graded_subcells_d = [None] * len(subcells_d)
    for i in d_order:
        
        # current cell width
        d = subcells_d[i]

        # previous cell width
        if (i > 0) and (graded_subcells_d[i - 1] is not None):
            dprev = graded_subcells_d[i - 1][-1]
        elif (i > 0):
            dprev = np.clip(subcells_d[i - 1], 0, d0[axis] )
        # If on the edge of the grid, match to the default d0
        else:
            dprev = d_bounds[axis][0]

        # next cell width
        if (i < len(subcells_d) - 1) and (graded_subcells_d[i + 1] is not None):
            dnext = graded_subcells_d[i + 1][0]
        elif (i < len(subcells_d) - 1) :
            dnext = np.clip(subcells_d[i + 1], 0, d0[axis] )
        # If on the edge of the grid, match to the boundary width
        else:
            dnext = d_bounds[axis][1]

        # if cell is bounded by d0 cells on either side, divide the cell equally 
        if all([(g / d0[axis] ) > 0.8 for g in [dprev, dnext, d]]):
            n_split = int(np.around(d / d0[axis] ))
            graded_subcells_d[i] = [d / n_split] * n_split
        # create a gradient of cell widths to span the space that minimizes the growth rate. 
        else:
            graded_subcells_d[i] = list(
                utils.blend_cell_widths(dprev, dnext, d, tol=self._tol)
            )

    # flatten list of lists of subcell widths
    mesh_cells_d[axis] = list(itertools.chain(*graded_subcells_d))


# Next, fill each mesh cell as either copper or not, for each layer along z. Loop through each object again and
# decide for each mesh cell? Seems like it can be vectorized and made reasonably fast, otherwise, do it in C++.
# Once each cell is filled, the e components can be assigned conductivities.






# Db_0 = s.dt / u0
# Cb_0 = s.dt / e0 
# p = s.plot_coefficients("ex_z", "a", "y", 0, point_size=15, cmap="brg")
# p.camera_position = "xy"
# p.show()

f0 = 10e9
pulse_n = 2800
# width of half pulse in time
pulse_width = (s.dt * 400)
# center of the pulse in time
t0 = (s.dt * 500)

vsrc = 1e-2 * self.gaussian_source(width=80e-12, t0=80e-12, t_len=500e-12)
plt.plot(vsrc)

frequency: np.ndarray = np.arange(5e9, 15e9, 10e6)

s.run([1], [vsrc], n_threads=4)

sdata = s.get_sparameters(frequency, downsample=False)
S11 = sdata[:, 0]
S21 = sdata[:, 1]


p = s.plot_monitor(["mon1"], el=0, zoom=1.1, az=0, view="xy", opacity=[0.8, 1], linear=False, cmap="jet", style="surface")
p.show(title="EM Solver")

# compute line impedance
line_i = self.vi_probe_values("c1")
line_v = self.vi_probe_values("v1")

IP = utils.dtft(self.vi_probe_values("c1"), frequency, 1 / s.dt)
VP = utils.dtft(-self.vi_probe_values("v1"), frequency, 1 / s.dt) * np.exp(1j * 2 * np.pi * frequency * (-s.dt / 2))
ZP = VP / IP
S11_zp = rfn.conv.gamma_z(ZP)

sdata_ref = line_ref.evaluate(frequency)["s"] 

fig, ax = plt.subplots()
plt.plot(frequency / 1e9, ZP.real)
ax.plot(frequency / 1e9, conv.z_gamma(S11))
plt.ylim([0, 120])
plt.axhline(y=z_ref, linestyle=":", color="k")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Impedance [Ohm]")
# mplm.line_marker(x = 10, axes=ax)
ax.legend(["probe", "port"])


fig, axes = plt.subplots(2, 2, figsize=(9, 9))

ax = axes[0,0]
rfn.plots.draw_smithchart(ax)
ax.plot(S11.real, S11.imag)
ax.plot(sdata_ref.sel(b=1, a=1).real, sdata_ref.sel(b=1, a=1).imag)



ax = axes[0,1]
ax.plot(frequency / 1e9, conv.db20_lin(S11))
ax.plot(frequency / 1e9, conv.db20_lin(sdata_ref).sel(b=1, a=1))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S11", "Ref"])

ax = axes[1,0]
ax.plot(frequency / 1e9, conv.db20_lin(S21))
ax.plot(frequency / 1e9, conv.db20_lin(sdata_ref).sel(b=2, a=1))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[dB]")
ax.legend(["S21", "Ref"])

ax = axes[1,1]
ax.plot(frequency / 1e9, np.unwrap(np.angle(S21, deg=True)))
ax.plot(frequency / 1e9, np.unwrap(np.angle(sdata_ref.sel(b=2, a=1), deg=True)))
mplm.line_marker(x = 10, axes=ax)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("[deg]")
ax.legend(["S21", "Ref"])


fig.tight_layout()
plt.show()
