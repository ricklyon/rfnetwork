

import rfnetwork as rfn

import numpy as np 
import matplotlib.pyplot as plt 
from rfnetwork import const, conv
import pyvista as pv
import unittest


class TestDipole(unittest.TestCase):
    """ 
    Test 10GHz dipole, closely parallels dipole.py example
    """
    def test_dipole(self):

        ms_w = 0.030
        # solve box size
        sbox_h = 1.1
        sbox_w = 1.1
        sbox_len = 1.5

        # gap between dipole legs
        gap = 0.015
        # end to end dipole length
        dipole_len = 0.546

        # edges of traces along y axis
        ms_y = (-ms_w / 2, ms_w / 2)

        # edges of traces along z axis
        ms1_z = (-(dipole_len / 2), -gap/2) 
        ms2_z = (gap / 2, (dipole_len / 2))

        sbox = pv.Cube(center=(0, 0, 0), x_length=sbox_len, y_length=sbox_w, z_length=sbox_h)

        ms_upper = pv.Rectangle([
            (0, ms_y[0], ms1_z[0]),
            (0, ms_y[1], ms1_z[0]),
            (0, ms_y[1], ms1_z[1])
        ])

        ms_lower = pv.Rectangle([
            (0, ms_y[0], ms2_z[0]),
            (0, ms_y[1], ms2_z[0]),
            (0, ms_y[1], ms2_z[1])
        ])

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
        s.generate_mesh(d0 = 0.03, d_edge=0.01)

        # setup wide-band far-field monitor
        s.add_farfield_monitor(frequency=[10e9, 20e9])

        # apply edge singularity correction to the edges of traces, iterate over lower leg and upper leg
        for i, ms_z in enumerate((ms1_z, ms2_z)):

            # left edge
            s.edge_correction(
                (0, ms_y[0], ms_z[0]), 
                (0, ms_y[0], ms_z[1]), 
                integration_line="y-"
            )

            # right edge
            s.edge_correction(
                (0, ms_y[1], ms_z[0]), 
                (0, ms_y[1], ms_z[1]), 
                integration_line="y+"
            )

            # top/lower edge
            s.edge_correction(
                (0, ms_y[0], ms_z[i]), 
                (0, ms_y[1], ms_z[i]), 
                integration_line=("z-" if i == 0 else "z+")
            )

        vsrc = s.gaussian_source(width=50e-12, t0=40e-12, t_len=600e-12)
        s.assign_excitation(vsrc, 1)
        s.solve(n_threads=4, show_progress=False)

        gain_db = conv.db10_lin(s.get_farfield_gain(theta=np.arange(0, 190, 10), phi=np.arange(-180, 184, 4)))

        thetapol_gain = gain_db.sel(frequency=10e9, polarization="thetapol", theta=90)
        phipol_gain = gain_db.sel(polarization="phipol")

        # check non-physical results above theoretical gain
        np.testing.assert_array_less(thetapol_gain, 2.15)

        # allow some numerical error up to 0.35dB since the course grid does not capture all the radiated power.
        np.testing.assert_array_less(np.abs(2.1 - thetapol_gain), 0.35)

        # check phipol is sufficiently attenuated
        np.testing.assert_array_less(phipol_gain, -40)

        # check null at pole
        np.testing.assert_array_less(gain_db.sel(theta=[0], polarization="thetapol"), -90)
        np.testing.assert_array_less(gain_db.sel(theta=[180], polarization="thetapol"), -90)

        # theta = gain_db.coords["theta"]
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(theta, gain_db.sel(frequency=10e9, phi=0, polarization="thetapol").squeeze(), label="10 GHz")
        # ax.plot(theta, gain_db.sel(frequency=20e9, phi=0, polarization="thetapol").squeeze(), label="20 GHz")
        # ax.set_xlabel(r"$\theta$ [deg]")
        # ax.set_ylim([-25, 5])

        # check S11 is less than -10dB
        s11 = s.get_sparameters([10e9], downsample=False)
        np.testing.assert_array_less(conv.db20_lin(s11), -10)

        # input impedance, not quite equal to theoretical value since it is a strip and not ideal wire.
        np.testing.assert_array_less(np.abs(73 - conv.z_gamma(s11).real), 3)


if __name__ == "__main__":
    unittest.main()