
import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import conv
import pyvista as pv
import rfnetwork as rfn
import mpl_markers as mplm
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage
import imageio.v2 as imageio
from skimage import filters
import io
from PIL import Image
from np_struct import ldarray
import skimage


# set matplotlib style
plt.style.use(rfn.DEFAULT_STYLE)
dir_ = Path(__file__).parent

pv.set_jupyter_backend("trame")
np.set_printoptions(suppress=True)
sys.argv = sys.argv[0:1]

bounding_box = pv.Box((-2.5, 2.5, -0.8, 0.8, -0.1, 5))

sub_h = 0.06
substrate = pv.Box((-2.0, 2.0, -sub_h, 0 , 0, 4))

s = rfn.FDTD_Solver(bounding_box)

s.add_dielectric(substrate, er=4.5, loss_tan=0.005, f0=3e9, style=dict(opacity=0.2))

s.add_image_layer(
    filepath = dir_ / "lab_project-B_Cu.gbr",
    origin = (-2.0197, -sub_h, -0.0197),
    width_axis = "x",
    length_axis = "z",
)

s.add_image_layer(
    filepath = dir_ / "lab_project-F_Cu.gbr",
    origin = (-1.83, 0, 0.175),
    width_axis = "x",
    length_axis = "z",
)

# # self = s
# top_img = s.gbr_images["lab_project-F_Cu"]["img"]
# btm_img = s.gbr_images["lab_project-B_Cu"]["img"]

port_x = 2
port_y0, port_y1 = (0.949, 1.059)
port_face = pv.Rectangle(((port_x, -sub_h, port_y0), (port_x, -sub_h, port_y1), (port_x, 0, port_y1)))
s.add_lumped_port(1, port_face, "y+")

# ensure ms trace extends all the way to the edge
trace_extension = pv.Rectangle(((port_x - 0.1, 0, port_y0), (port_x, 0, port_y0), (port_x, 0, port_y1)))

s.add_conductor(trace_extension)
s.assign_PML_boundaries("x-", "x+", "y-", "y+", "z+", n_pml=5)
s.generate_mesh(0.06, 0.005)

p = s.render()

s.Nx * s.Ny * s.Nz / 1e3
s.plot_coefficients("ez_x", "b", "y", 0).show()
# s.plot_coefficients("ez_x", "b", "y", -sub_h).show()

s.add_field_monitor("mon1", "e_total", axis="y", position=-sub_h, n_step=10)
s.add_farfield_monitor([2.2e9, 4e9])

vsrc = s.gaussian_modulated_source(3e9, width=500e-12, t0=300e-12, t_len=1000e-12)
plt.plot(vsrc)

s.assign_excitation(vsrc, 1)
s.solve(n_threads=4)

frequency = np.arange(0.01, 6, 0.01) * 1e9
sdata = s.get_sparameters(frequency, downsample=False)
S11 = sdata[:, 0]

fig, ax = plt.subplots()
ax.plot(frequency / 1e6, conv.db20_lin(S11))
ax.set_ylim([-20, 5])
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("[dB]")


theta_cut = rfn.conv.db10_lin(
    s.get_farfield_gain(theta=np.arange(-180, 181, 2), phi=0).sel(polarization="thetapol")
)

fig1, ax1 = plt.subplots(subplot_kw=dict(projection="polar"))

theta_rad = np.deg2rad(theta_cut.coords["theta"])

ax1.plot(theta_rad, theta_cut.squeeze().T)

for ax in (ax1,):
    ax.set_theta_zero_location('N') 
    ax.set_theta_direction(-1) 
    ax.set_ylim([-25, 10])
    ax.set_yticks(np.arange(-25, 15, 5))
    ax.set_yticklabels(["", "-15", "-10", "-5", "0", "5", "10dBi"])

    # Set theta labels
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    labels = [f"{d}°" for d in [0, 45, 90, 135, 180, -135, -90, -45]]
    ax.set_xticklabels(labels)

ax1.set_xlabel(r"$\theta$ [deg], $\phi$=0°")
ax1.legend(["{:.3f}GHz".format(f/1e9) for f in theta_cut.coords["frequency"]])
plt.show()


cpos = pv.CameraPosition(
    position=(1, -0.3, 0.0),
    focal_point=(0, 0, 0.00),
    viewup=(0, 0.0, 1.0),
)

# gif_setup = dict(file = dir_ / "flare.gif", fps=15, step_ps=5)
s.plot_monitor("mon1", opacity=0.8, camera_position=cpos).show()
