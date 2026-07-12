"""
Profile solver performance
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pyvista as pv

import rfnetwork as rfn

from timeit import timeit
import unittest
import pytest

# set matplotlib style
plt.style.use(rfn.DEFAULT_STYLE)

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
        s.assign_PML_boundaries("x-", "x+", "y-", "y+", "z+", "z-", n_pml=5)

        def time_solve(
            d_min: float = 0.01, d_max: float = 0.02, t_len: float = 600e-12, gpu: bool = False, iterations: int = 3
        ):
            """
            Report time of solver with the given grid settings.
            """
            s.generate_mesh(d_max = d_max, d_min=d_min)
            s.add_farfield_monitor(frequency=10e9)

            vsrc = s.gaussian_source(width=50e-12, t0=40e-12, t_len=t_len)
            s.reset_excitations()
            s.assign_excitation(vsrc, 1)

            if gpu:
                total_time = timeit("s.solve(show_progress=False, gpu=True)", number=iterations, globals=globals())
            else:
                total_time = timeit("s.solve(show_progress=False, gpu=False)", number=iterations, globals=globals())

            dev = "CPU" if not gpu else "GPU"
            print(f"Cells: {s.Nx * s.Ny * s.Nz / 1e3}k. Time Steps: {len(s.time)}. {dev} Time: {total_time / iterations:.2f}s")


        def plot():
            """
            Plot dipole response.
            """
            import time
            stime = time.time()
            pp_gain = rfn.conv.db10_lin(
                s.get_farfield_gain(theta=np.arange(-180, 181, 1), phi=[0, 90]).sel(polarization="thetapol")
            )
            print(time.time() - stime)

            fig, (ax) = plt.subplots(1, 1, subplot_kw=dict(projection="polar"), figsize=(8, 4))

            theta_rad = np.deg2rad(pp_gain.coords["theta"])

            ax.plot(theta_rad, pp_gain.squeeze(), label=f"{10e9/1e9:.0f} GHz")

            ax.set_theta_zero_location('N') 
            ax.set_theta_direction(-1) 
            ax.set_xlabel(r"$\theta$ [deg], $\phi$=0°")
            ax.set_ylim([-25, 5])
            ax.set_yticks(np.arange(-25, 10, 5))
            ax.set_yticklabels(["", "-20", "-15", "10", "-5", "0", "5dBi"])
            ax.legend(loc="lower right")

            # Set theta labels
            ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            labels = [f"{d}°" for d in [0, 45, 90, 135, 180, -135, -90, -45]]
            ax.set_xticklabels(labels)

            fig.tight_layout()

            frequency: np.ndarray = np.arange(0, 40.01e9, 10e6)
            sdata_raw = s.get_sparameters(frequency, downsample=False)
            # cast as component to use plot functions
            sdata = rfn.Component_Data(sdata_raw)

            sdata.plot(11, fmt="db")


        # TODO: fix gpu solver error on second run
        # 1253.616k
        time_solve(d_max=0.01, d_min=0.005, t_len=400e-12, gpu=False, iterations=1)
        time_solve(d_max=0.01, d_min=0.005, t_len=400e-12, gpu=True, iterations=1)

        # 156.8k
        time_solve(gpu=False, iterations=1)
        time_solve(gpu=True, iterations=1)

        # plot()


if __name__ == "__main__":
    unittest.main()