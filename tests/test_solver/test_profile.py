"""
Profile solver performance
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pyvista as pv

import rfnetwork as rfn
from rfnetwork import conv

from timeit import timeit
import unittest
import pytest
from pathlib import Path
from np_struct import ldarray

import time


DATA_DIR = Path(__file__).parent.parent / "data"

class TestDipoleProf(unittest.TestCase):
    """ 
    Test 10GHz dipole, closely parallels dipole.py example
    """
    
    @pytest.mark.skip()
    def test_dipole_prof(self):
        # trace width
        ms_w = 0.030

        # solve box size
        sbox_h = 1.2
        sbox_w = 1.0
        sbox_len = 1.0

        # gap between dipole legs
        gap = 0.015
        # end to end dipole length
        dipole_len = 0.546

        # %%
        # Build Dipole Model
        # ------------------------

        # edges of traces along y axis
        ms_y = (-ms_w / 2, ms_w / 2)

        # edges of traces along z axis
        ms1_z = (-(dipole_len / 2), -gap/2) 
        ms2_z = (gap / 2, (dipole_len / 2))

        # solve box
        sbox = pv.Cube(center=(0, 0, 0), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

        # upper leg of dipole
        ms_upper = pv.Rectangle([
            (0, ms_y[0], ms1_z[0]),
            (0, ms_y[1], ms1_z[0]),
            (0, ms_y[1], ms1_z[1])
        ])

        # lower leg
        ms_lower = pv.Rectangle([
            (0, ms_y[0], ms2_z[0]),
            (0, ms_y[1], ms2_z[0]),
            (0, ms_y[1], ms2_z[1])
        ])

        # port between upper and lower leg
        port1_face = pv.Rectangle([
            (0, ms_y[0], gap/2),
            (0, ms_y[1], gap/2),
            (0, ms_y[1], -gap/2)
        ])

        s = rfn.FDTD_Solver(sbox)
        s.add_conductor(ms_upper, ms_lower, style=dict(color="gold"))
        s.add_lumped_port(1, port1_face, "z-")

        # PML boundaries are required on all sides to add a far-field monitor
        s.add_PML("x-", "x+", "y-", "y+", "z+", "z-", n_pml=5)

        def time_solve(
            s: rfn.FDTD_Solver, 
            d_min: float = 0.01, 
            d_max: float = 0.02, 
            t_len: float = 400e-12, 
            gpu: bool = False, 
            iterations: int = 3
        ) -> ldarray:
            """
            Report time of solver with the given grid settings.
            """
            s.generate_mesh(d_max = d_max, d_min=d_min)
            s.add_farfield_monitor(frequency=10e9)

            vsrc = s.gaussian_source(width=50e-12, t0=40e-12, t_len=t_len)
            s.reset_excitations()
            s.assign_excitation(vsrc, 1)

            if gpu:
                total_time = timeit("s.solve(show_progress=False, gpu=True)", number=iterations, globals=locals())
            else:
                total_time = timeit("s.solve(show_progress=False, gpu=False)", number=iterations, globals=locals())

            dev = "CPU" if not gpu else "GPU"
            avg_time = total_time / iterations
            print(f"Cells: {s.Nx * s.Ny * s.Nz / 1e3}k. Time Steps: {len(s.time)}. {dev} Time: {avg_time:.2f}s")

            return rfn.conv.db20_lin(
                s.get_farfield_gain(theta=np.arange(20, 161, 1), phi=[0, 90]).sel(polarization="thetapol")
            )

        # 432k cells
        gain_cpu = time_solve(s, d_max=0.015, d_min=0.005, iterations=1)
        # gain_gpu = time_solve(s, d_max=0.015, d_min=0.005, iterations=1, gpu=True)

        # gain_cpu.save(DATA_DIR / f"regression/test_dipole_prof_1.npy")
        gain_ref = ldarray.load(DATA_DIR / f"regression/test_dipole_prof_1.npy")

        # np.testing.assert_array_almost_equal(gain_cpu, gain_ref, decimal=2)
        # np.testing.assert_array_almost_equal(gain_gpu, gain_ref, decimal=2)

        # 156k cells
        gain_cpu = time_solve(s, d_max=0.02, d_min=0.01, iterations=3)
        # gain_gpu = time_solve(s, d_max=0.02, d_min=0.01, iterations=3, gpu=True)

        # gain_cpu.save(DATA_DIR / f"regression/test_dipole_prof_2.npy")
        gain_ref = ldarray.load(DATA_DIR / f"regression/test_dipole_prof_2.npy")

        # np.testing.assert_array_almost_equal(gain_cpu, gain_ref, decimal=2)
        # np.testing.assert_array_almost_equal(gain_gpu, gain_ref, decimal=2)

        # pp_gain = rfn.conv.db20_lin(
        #     s.get_farfield_gain(theta=np.arange(-180, 181, 1), phi=[0]).sel(polarization="thetapol")
        # )

        # fig, (ax) = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(8, 4))

        # theta_rad = np.deg2rad(pp_gain.coords["theta"])
        # ax.plot(theta_rad, pp_gain.squeeze())

        # ax.set_theta_zero_location('N') 
        # ax.set_theta_direction(-1) 
        # ax.set_xlabel(r"$\theta$ [deg], $\phi$=0°")
        # ax.set_ylim([-25, 5])
        # ax.set_yticks(np.arange(-25, 10, 5))
        # ax.set_yticklabels(["", "-20", "-15", "10", "-5", "0", "5dBi"])
        # ax.legend(loc="lower right")

        # # Set theta labels
        # ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))

    def profile_patch(self):
        solve_w = conv.in_mm(35)
        solve_len = conv.in_mm(38)
        solve_h = conv.in_mm(10)
        sbox = pv.Cube(center=(0, 0, solve_h / 2), x_length=solve_w, y_length=solve_len, z_length=solve_h)

        sub_h = 0.039
        feed_w = conv.in_mm(1.9)
        feed_len = conv.in_mm(9) # from center
        inset_len = conv.in_mm(4)
        inset_w = conv.in_mm(1)
        patch_w = conv.in_mm(16)
        patch_len = conv.in_mm(12)


        s = rfn.FDTD_Solver(sbox)
        feed = pv.Rectangle([(-feed_w / 2, 0, sub_h), (feed_w / 2, 0, sub_h), (feed_w / 2, -feed_len, sub_h)])

        patch_left = pv.Rectangle(
            [(-patch_w / 2, -patch_len / 2, sub_h), (-patch_w / 2, patch_len / 2, sub_h), (-inset_w / 2 - feed_w / 2, patch_len / 2, sub_h)]
        )

        patch_right = pv.Rectangle(
            [(patch_w / 2, -patch_len / 2, sub_h), (patch_w / 2, patch_len / 2, sub_h), (inset_w / 2 + feed_w / 2, patch_len / 2, sub_h)]
        )

        patch_top = pv.Rectangle(
            [(-patch_w / 2, -patch_len / 2 + inset_len, sub_h), (-patch_w / 2, patch_len / 2, sub_h), (patch_w / 2, patch_len / 2, sub_h)]
        )

        substrate = pv.Cube(center=(0, 0, sub_h / 2), x_length=solve_w, y_length=solve_len, z_length=sub_h)

        s.add_conductor(feed, patch_left, patch_right, patch_top, style=dict(color="gold"))
        s.add_dielectric(substrate, er=4.3, loss_tan=0.02, f0 = 5e9, style=dict(opacity=0.5))

        # port between upper and lower leg
        port1_face = pv.Rectangle([
            (-feed_w / 2, -feed_len, 0),
            (feed_w / 2, -feed_len, 0),
            (feed_w / 2, -feed_len, sub_h)
        ])

        s.add_lumped_port(1, port1_face, "z-")

        # PML boundaries are required on all sides to add a far-field monitor
        s.add_PML("x-", "x+", "y-", "y+", "z+", n_pml=5)
        s.generate_mesh(d_max = 0.015, d_min=0.005)

        vsrc = s.gaussian_source(width=100e-12, t0=60e-12, t_len=2000e-12)
        s.assign_excitation(vsrc, 1)

        stime = time.time()
        s.solve(n_threads=4)
        print(f"Cells: {s.Nx * s.Ny * s.Nz / 1e3}k. Time Steps: {len(s.time)}. {dev} Time: {avg_time:.2f}s")

        frequency: np.ndarray = np.arange(5e9, 6.4e9, 1e6)
        sdata_raw = s.get_sparameters(frequency, downsample=False)
        # cast as component to use plot functions
        sdata = rfn.Component_Data(sdata_raw)

        sdata.plot(11, fmt="db")
        plt.show()


                                
if __name__ == "__main__":
    unittest.main()