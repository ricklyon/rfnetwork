import pyvista as pv
import unittest
from parameterized import parameterized

from np_struct import ldarray
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rfnetwork as rfn

from rfnetwork import const, conv, antennas, utils

class TestSphereRCS(unittest.TestCase):

    @parameterized.expand(
        list(np.logspace(-2, 1, 41))
    )
    def test_plate_rcs(self, r_lam: float):
        """
        Test RCS of a metallic sphere.

        Parameters
        ----------
        r_lam: float
            radius of sphere normalized by the wavelength

        """
        frequency = 3e9
        lam0 = rfn.const.c0 / frequency
        beta = 2 * np.pi / lam0

        # isotropic source antenna at the origin
        az, el = np.arange(-90, 92, 2), np.arange(-90, 92, 2)
        tx_patt = ldarray(
            np.zeros((3, 1, len(az), len(el))),
            coords=dict(polarization=("x", "y", "z"), frequency=frequency, az=az, el=el)
        )
        tx_patt[dict(polarization="y")] = 1

        src_patt = tx_patt
        scatterer1_loc = (0, 0, 500)
        src_pwr = 1
        rx_pos = np.array((0, 0, 0))
        src_pos = np.array((0, 0, 0))
        rx_patt = tx_patt.copy()

        # reset sphere to unit size
        res = np.clip(int(2 * np.pi * r_lam * 5), 60, 500)
        obj = pv.Sphere(radius=r_lam * lam0, theta_resolution=res, phi_resolution=res)

        # move to desired location
        obj.points += scatterer1_loc

        Js, Ms, _ = antennas.illuminate_object(obj, src_pos, src_patt, src_pwr, frequency, spherical=True)

        e_rx_terminal, _ = antennas.get_scattering_rxE(obj, Js, Ms, rx_patt, rx_pos, frequency)

        rcs_actual = antennas.monostatic_rcs(
            obj, src_pos, src_pwr, e_rx_terminal, frequency, tx_patt, rx_patt
        ).sel(polarization="y")

        # theta = np.arange(-180, 182, 2)
        # rE_local_phi0 = antennas.get_scattering_rE(scatterer1, Js, Ms, theta, np.ones_like(theta) * 0, frequency).squeeze()
        # rE_local_phi0.coords["phitheta"] = theta
        # rE_local_phi0 = rE_local_phi0.rename(phitheta="theta")

        # fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        # rE_local_phi0.plot("theta", xfmt=np.deg2rad, yfmt="db20", polarization="phipol")
        # plt.show()

        # plot normalized rcs
        ka = 2 * np.pi * r_lam 

        ref_rcs = utils.sphere_mie_monostatic(r_lam * lam0, frequency)
        err = np.abs(conv.db10_lin(rcs_actual / ref_rcs))

        # check RCS is correct up to half a dB
        np.testing.assert_array_less(err, 0.5)

if __name__ == "__main__":
    unittest.main()