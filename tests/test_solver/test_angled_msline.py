import numpy as np 
from rfnetwork import const, conv
import pyvista as pv

import rfnetwork as rfn

from scipy.spatial.transform import Rotation

from parameterized import parameterized
import unittest

class TestAngledLine(unittest.TestCase):

    @parameterized.expand([
        30, 60 # angle of microstrip line from the x axis, degrees
    ])
    def test_angled_line(self, ms_ang_deg):
        
        # microstrip width
        ms_w = 0.04
        # substrate height
        sub_h = 0.02   
        # straight section length
        ms_len0 = 0.15   
        # angled section length
        ms_len1 = 0.4

        ms_len1_x = ms_len1 * np.cos(np.deg2rad(ms_ang_deg))
        ms_len1_y = ms_len1 * np.sin(np.deg2rad(ms_ang_deg))
        ms_x = ((-ms_len1_x / 2), (ms_len1_x / 2))
        ms_y = (-ms_len1_y / 2, ms_len1_y / 2)

        # solve box size
        sbox_h = 0.35
        sbox_w = ms_len1_x + 0.5
        sbox_len = ms_len1_y + 0.5

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
        rot = Rotation.from_euler("z", ms_ang_deg, degrees=True).as_matrix()
        ms_ang_trace.points = np.einsum("ij,kj->ki", rot, ms_ang_trace.points)
        # translate to position
        ms_ang_trace.points += (ms_x[0], ms_y[0], 0)

        # corners
        corner0 = pv.Triangle([ms_ang_trace.points[0], (ms_x[0], ms_y[0], sub_h), (ms_x[0], ms_y[0] - ms_w/2, sub_h)])
        corner1 = pv.Triangle([ms_ang_trace.points[2], (ms_x[1], ms_y[1], sub_h), (ms_x[1], ms_y[1] + ms_w/2, sub_h)])

        # add ports
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


        s = rfn.FDTD_Solver(sbox)
        s.add_dielectric(substrate, er=3.66, style=dict(opacity=0.0))
        s.add_conductor(ms0_trace, style=dict(color="gold"))
        s.add_conductor(ms1_trace, style=dict(color="gold"))
        s.add_conductor(ms_ang_trace, style=dict(color="gold"))
        s.add_conductor(corner0, corner1, style=dict(color="gold"))
        s.add_lumped_port(1, port1_face, "z+")
        s.add_lumped_port(2, port2_face, "z+")

        s.assign_PML_boundaries("z+", "y-", "y+", n_pml=5)
        s.generate_mesh(d_max = 0.02, d_min=0.007)

        # rough check on mesh resolution
        self.assertGreater(s.Nx * s.Ny * s.Nz / 1e3, 90)
        self.assertLess(s.Nx * s.Ny * s.Nz / 1e3, 150)

        # p = s.plot_coefficients("ex_y", "b", "z", sub_h, point_size=15, cmap="brg")
        # p.camera_position = "xy"
        # p.show()

        vsrc = 1e-2 * s.gaussian_source(width=80e-12, t0=80e-12, t_len=400e-12)
        frequency: np.ndarray = np.arange(6e9, 12e9, 10e6)

        s.assign_excitation(vsrc, 1)
        s.solve(n_threads=4, show_progress=False)

        sdata = s.get_sparameters(frequency, downsample=False)

        # check S11 < -20dB
        np.testing.assert_array_less(conv.db20_lin(sdata.sel(b=1, a=1)), -20)
        # check S21 < 0dB, S21 > 0.2dB
        np.testing.assert_array_less(conv.db20_lin(sdata.sel(b=2, a=1)), 0)
        np.testing.assert_array_less(conv.db20_lin(1  / sdata.sel(b=2, a=1)), 0.2)

        # fig, axes = plt.subplots(2, 2, figsize=(9, 9))

        # ax = axes[0,1]
        # ax.plot(frequency / 1e9, conv.db20_lin(sdata.sel(b=1, a=1)))
        # ax.set_xlabel("Frequency [GHz]")
        # ax.set_ylabel("[dB]")
        # ax.legend(["S11", "Ref"])

        # ax = axes[1,0]
        # ax.plot(frequency / 1e9, conv.db20_lin(sdata.sel(b=2, a=1)))
        # ax.set_xlabel("Frequency [GHz]")
        # ax.set_ylabel("[dB]")
        # ax.legend(["S21", "Ref"])

        # ax = axes[1,1]
        # ax.plot(frequency / 1e9, np.unwrap(np.angle(sdata.sel(b=2, a=1), deg=True)))
        # ax.set_xlabel("Frequency [GHz]")
        # ax.set_ylabel("[deg]")
        # ax.legend(["S21", "Ref"])

        # fig.tight_layout()
        # plt.show()


if __name__ == "__main__":
    unittest.main()