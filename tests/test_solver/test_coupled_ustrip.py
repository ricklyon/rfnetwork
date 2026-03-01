
import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv, utils
import pyvista as pv
import rfnetwork as rfn

import unittest
from parameterized import parameterized
from itertools import permutations


class TestCoupledUStrip(unittest.TestCase):

    # reference values taken from this online calculator
    # https://wcalc.sourceforge.net/cgi-bin/coupled_microstrip.cgi
    # width, space, odd mode z reference, even mode z reference
    @parameterized.expand([
        (0.02, 0.01, 60.3447, 120.393),
        (0.03, 0.005, 45.0888, 101.847), 
        (0.06, 0.01, 38.674, 63.9801),
    ])
    def test_ustrip_axis(self, ms_w, ms_sp, ref_odd, ref_even):

        frequency = np.arange(5e9, 15e9, 10e6)
        f0 = 10e9

        sub_h = 0.03  # substrate height
        er =  3.66  # relative permittivity

        # solve box dimensions, inches
        sbox_h = 0.25
        sbox_w = 0.3
        sbox_len = 0.25

        # center locations of microstrip lines along y axis
        ms1_y = -(ms_w / 2) - (ms_sp / 2)
        ms2_y = (ms_w / 2) + (ms_sp / 2)

        # end locations of lines along x axis, lines terminate in PML region 
        ms_x = (-sbox_len/2 + 0.1, sbox_len/2)

        # substrate geometry
        substrate = pv.Cube(
            center=(0, 0, sub_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sub_h
        )

        # solve box
        sbox = pv.Cube(
            center=(0, 0, sbox_h/2), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h
        )

        # Create 3D Model
        s = rfn.FDTD_Solver(sbox)
        s.add_dielectric(substrate, er=er, style=dict(opacity=0.2))

        # add microstrip lines
        for i, ms_y in enumerate((ms1_y, ms2_y)):
            ms_trace = pv.Rectangle([
                (ms_x[0], ms_y - ms_w/2, sub_h),
                (ms_x[0], ms_y + ms_w/2, sub_h),
                (ms_x[1], ms_y + ms_w/2, sub_h)
            ])
            s.add_conductor(ms_trace, style=dict(color="gold"))

            # add lumped ports
            port_face = pv.Rectangle([
                (ms_x[0], ms_y - ms_w/2, sub_h),
                (ms_x[0], ms_y + ms_w/2, sub_h),
                (ms_x[0], ms_y + ms_w/2, 0),
            ])
            s.add_lumped_port(i+1, port_face, integration_axis="z-")

        # assign PML layers, omitting the x- side near the ports
        s.assign_PML_boundaries("x+", "z+", "y-", "y+", n_pml=5)

        # create mesh with a nominal width of 20mils far from geometry edges, and 2.5mils near edges.
        # cell widths are tapered to minimize errors
        s.generate_mesh(d0 = 0.02, d_edge = 0.0025)

        # apply edge singularity correction to the edges along the length of the microstrip lines
        for i, ms_y in enumerate((ms1_y, ms2_y)):
            p1 = (ms_x[0], ms_y + ms_w/2, sub_h)
            p2 = (ms_x[1], ms_y + ms_w/2, sub_h)

            s.edge_correction(p1, p2, integration_axis="y+")

            p1 = (ms_x[0], ms_y - ms_w/2, sub_h)
            p2 = (ms_x[1], ms_y - ms_w/2, sub_h)

            s.edge_correction(p1, p2, integration_axis="y-")

        # create voltage waveform. Time units are in seconds
        vsrc = 1e-2 * s.gaussian_modulated_source(f0=10e9, width=200e-12, t0=160e-12, t_len=400e-12)

        # run even mode, 
        # same waveform at both port 1 and 2
        s.assign_excitation(vsrc, [1, 2])
        s.solve(n_threads=4, show_progress=False)
        S_even = s.get_sparameters(frequency)

        # Run Odd Mode 
        # setup opposite polarity waveforms at each port
        s.assign_excitation(vsrc, 1)
        s.assign_excitation(-vsrc, 2)
        s.solve(n_threads=4, show_progress=False)
        S_odd = s.get_sparameters(frequency)

        # compute even/odd impedances
        z_odd = conv.z_gamma(S_odd.sel(b=1, a=1, frequency=f0)).real
        z_even = conv.z_gamma(S_even.sel(b=1, a=1, frequency=f0)).real
        
        # fig, ax = plt.subplots()
        # ax.plot(frequency / 1e9, conv.z_gamma(S_odd.sel(b=1, a=1)).real)
        # ax.plot(frequency / 1e9, conv.z_gamma(S_even.sel(b=1, a=1)).real)

        # ax.set_ylim([0, 110])
        # ax.axhline(y=ref_odd, linestyle=":", color="tab:blue")
        # ax.axhline(y=ref_even, linestyle=":", color="tab:orange")
        # ax.set_xlabel("Frequency [GHz]")
        # ax.set_ylabel("Impedance [Ohm]")
        # ax.legend(["Even Mode", "Odd Mode", "Ref Odd", "Ref Even"])
        # ax.set_title("Odd/Even Impedance of Coupled Microstrip")

        # plt.show()

        # verify impedances are within 1.5 ohms of reference
        np.testing.assert_array_less(
            np.abs(z_odd - ref_odd), 1.5
        )
        np.testing.assert_array_less(
            np.abs(z_even - ref_even), 1.5
        )


if __name__ == "__main__":
    unittest.main()