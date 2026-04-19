
import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import rfnetwork as rfn

import unittest
from parameterized import parameterized
from itertools import permutations


class TestCoupledStripline(unittest.TestCase):

    # reference values taken from this online calculator
    @parameterized.expand([
        (0.02, 0.01),
        (0.03, 0.005), 
        (0.04, 0.01),
    ])
    def test_ustrip_axis(self, sl_w, sl_sp):
        
        frequency = np.arange(5e9, 15e9, 10e6)
        f0 = 10e9

        b = 0.06  # substrate height
        er =  3.66  # relative permittivity

        # solve box dimensions, inches
        sbox_w = 0.4
        sbox_len = 0.25

        # center locations of lines along y axis
        line1_y = -(sl_w / 2) - (sl_sp / 2)
        line2_y = (sl_w / 2) + (sl_sp / 2)

        # end locations of lines along x axis, lines terminate in PML region 
        ms_x = (-sbox_len/2 + 0.1, sbox_len/2)

        # substrate geometry
        substrate = pv.Cube(
            center=(0, 0, 0), x_length=sbox_len, y_length=sbox_w, z_length=b
        )

        # solve box
        sbox = pv.Cube(
            center=(0, 0, 0), x_length=sbox_len, y_length=sbox_w, z_length=b
        )

        s = rfn.FDTD_Solver(sbox)
        s.add_dielectric(substrate, er=er, style=dict(opacity=0.0))

        # add lines
        for i, y in enumerate((line1_y, line2_y)):
            ms_trace = pv.Rectangle([
                (ms_x[0], y - sl_w/2, 0),
                (ms_x[0], y + sl_w/2, 0),
                (ms_x[1], y + sl_w/2, 0)
            ])
            s.add_conductor(ms_trace, style=dict(color="gold"))

            # add lumped ports
            port_face = pv.Rectangle([
                (ms_x[0], y - sl_w/2, -b/2),
                (ms_x[0], y + sl_w/2, -b/2),
                (ms_x[0], y + sl_w/2, b/2),
            ])

            integration_line = pv.Line((ms_x[0], y, -b/2), (ms_x[0], y, 0))
            s.add_lumped_port(i + 1, port_face, integration_line=integration_line)


        # assign PML layers, omitting the x- side near the ports
        s.assign_PML_boundaries("x+", n_pml=5)

        # create mesh with a nominal width of 20mils far from geometry edges, and 2.5mils near edges.
        # cell widths are tapered to minimize errors
        s.generate_mesh(d_max = 0.01, d_min = 0.0025)

        # apply edge singularity correction to the edges along the length of the microstrip lines
        for i, y in enumerate((line1_y, line1_y)):
            p1 = (ms_x[0], y + sl_w/2, 0)
            p2 = (ms_x[1], y + sl_w/2, 0)

            s.edge_correction(p1, p2, integration_line="y+")

            p1 = (ms_x[0], y - sl_w/2, 0)
            p2 = (ms_x[1], y - sl_w/2, 0)

            s.edge_correction(p1, p2, integration_line="y-")


        # create voltage waveform. Time units are in seconds
        vsrc = 1e-2 * s.gaussian_modulated_source(f0=10e9, width=200e-12, t0=100e-12, t_len=500e-12)

        # run even mode, 
        # same waveform at both port 1 and 2
        s.assign_excitation(vsrc, [1, 2])
        s.solve(n_threads=4, show_progress=False)

        S_even = s.get_sparameters(frequency)

        # Run Odd Mode 
        # setup opposite polarity waveforms at each port
        s.reset_excitations()
        s.assign_excitation(vsrc, 1)
        s.assign_excitation(-vsrc, 2)
        s.solve(n_threads=4, show_progress=False)
        S_odd = s.get_sparameters(frequency)

        # reference impedance
        ref_odd, ref_even = rfn.utils.coupled_sline_impedance(sl_w, sl_sp, b, er)

        # compute even/odd impedances
        z_odd = conv.z_gamma(S_odd.sel(b=1, frequency=f0)).real
        z_even = conv.z_gamma(S_even.sel(b=1, frequency=f0)).real
        
        # fig, ax = plt.subplots()
        # ax.plot(frequency / 1e9, conv.z_gamma(S_odd.sel(b=1)).real, color="tab:orange")
        # ax.plot(frequency / 1e9, conv.z_gamma(S_even.sel(b=1)).real, color="tab:blue")

        # ax.set_ylim([0, 110])
        # ax.axhline(y=z_odd, linestyle=":", color="tab:orange")
        # ax.axhline(y=z_even, linestyle=":", color="tab:blue")
        # ax.set_xlabel("Frequency [GHz]")
        # ax.set_ylabel("Impedance [Ohm]")
        # ax.legend(["Odd Mode", "Even Mode", "Ref Odd", "Ref Even"])
        # # mplm.line_marker(x = 10, axes=ax)
        # ax.set_title("Odd/Even Impedance of Coupled Stripline")
        # plt.show()

        # verify impedances are within 2 ohms of reference
        np.testing.assert_array_less(
            np.abs(z_odd - ref_odd), 2
        )
        np.testing.assert_array_less(
            np.abs(z_even - ref_even), 2
        )


if __name__ == "__main__":
    unittest.main()