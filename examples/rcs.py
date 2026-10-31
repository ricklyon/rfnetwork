import pyvista as pv
import vtk
from np_struct import ldarray
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rfnetwork as rfn

from pathlib import Path

dir_ = Path(__file__).parent


f0 = 3e9
lam0 = rfn.const.c0 / f0
beta = 2 * np.pi / lam0

pv.set_jupyter_backend("trame")

# TX beam pattern, x-polarization
tx_patt = ldarray.load(dir_ / "data/vivaldi_antenna_patt.npy")
tx_pin = 1 # input power in watts

# move the tx pattern to the source position, currently centered at the origin
antenna_height = 1.5
source_pos = (0, antenna_height, 0)
src_patt = rfn.antennas.translate(tx_patt, source_pos)

# rx antenna patterns
rx_pos = (0, antenna_height - 0.1, 0)
rx_patt = rfn.antennas.translate(tx_patt, rx_pos)

# add antenna rendering
antenna = pv.read(dir_ / "data/horn_antenna.stl")
antenna.points *= 0.1
antenna.points += source_pos

scatterer = pv.Sphere(radius=0.4, theta_resolution=100, phi_resolution=100)

# move to desired location
scatterer.points += (-1, 0.8, 5)

J_xyz_dir, M_xyz_dir, _ = rfn.antennas.illuminate_object(scatterer, source_pos, tx_patt, 1, f0)
J_xyz_im, M_xyz_im, _ = rfn.antennas.illuminate_object(scatterer, source_pos, tx_patt, 1, f0, image=True)

J_xyz = J_xyz_dir + J_xyz_im
M_xyz = M_xyz_dir + M_xyz_im

plotter = pv.Plotter()
scatterer.cell_data["Js"] = np.linalg.norm(J_xyz, axis=-1) 
scatterer.cell_data["Ms"] = np.linalg.norm(M_xyz, axis=-1) 

# ground plane
gnd = pv.Rectangle(((-2, 0, 0), (2, 0, 0), (2, 0, 6)))

plotter.add_mesh(antenna, color="tan")
plotter.add_mesh(gnd, color="silver", metallic=1, specular=1.0)
plotter.add_mesh(scatterer, cmap="jet", scalars="Ms", clim=(0, 6))
plotter.add_axes()
plotter.show_bounds(bounds=(-2, 2, 0, 3, 0, 6), show_zlabels=False, show_xlabels=False, show_ylabels=False, xtitle=" ", ytitle=" ", ztitle=" ")
plotter.add_bounding_box()

cpos = pv.CameraPosition(position=(-9.2541, 4.4767, -3.2586), focal_point=(-0.1153, 1.0408, 2.8409), viewup=(0, 1, 0))
plotter.camera_position = cpos
plotter.show()

theta = np.arange(-180, 182, 2)
phi = np.zeros_like(theta)

rE_local = rfn.antennas.get_scattering_rE(scatterer, J_xyz, M_xyz, theta, phi, f0)
rE_local = rE_local.rename(phitheta="theta")
rE_local.coords["theta"] = theta

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
rE_local.plot("theta", xfmt=np.deg2rad, yfmt="db20")
plt.show()

# rx antenna pattern
rx_pos = [
    (-0.1, antenna_height, 0), 
    (0.1, antenna_height, 0),
]
rx_pos = ldarray(rx_pos, coords=dict(ele=range(2), axis=("x", "y", "z")))

rx_patt = rfn.antennas.translate(tx_patt, rx_pos)

rx_e, _ = rfn.antennas.get_scattering_rxE(scatterer, J_xyz, M_xyz, rx_patt, rx_pos, f0)
