import pyvista as pv
import unittest
from parameterized import parameterized

from np_struct import ldarray
import numpy as np

import matplotlib.pyplot as plt
from rfnetwork import antennas, conv, const


class TestPlateRCS(unittest.TestCase):

    @parameterized.expand(
        list(np.logspace(-1, 1, 21))
    )
    def test_plate_rcs(self, d_lam: float):
        """
        Test RCS of a flat plate. Expected RCS is (4pi / lam^2) * A

        Parameters
        ----------
        d_lam: float
            width of square plate normalized by the wavelength
        """

        frequency = 3e9
        lam0 = const.c0 / frequency

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

        obj = pv.Rectangle(
            ((-1.0, -1, 0), (-1, 1, 0), (1, 1, 0))
        )
        # scale to desired size
        obj.points *= ((d_lam * lam0)/2)

        # subdivide to 10 cells per wavelength
        # 10 = lam0 / (d_lam * lam0)
        # each subdivide creates 4 triangles
        subd = int(np.clip((d_lam * 5)/4, 5, 8))
        obj = obj.triangulate().subdivide(subd, subfilter='linear')

        # move to desired location
        obj.points += scatterer1_loc

        Js, Ms = antennas.illuminate_object(obj, src_pos, src_patt, src_pwr, frequency, spherical=False)

        e_rx_terminal = antennas.get_scattering_rxE(obj, Js, Ms, rx_patt, rx_pos, frequency)

        rcs_actual = antennas.monostatic_rcs(
            obj, src_pos, src_pwr, e_rx_terminal, frequency, tx_patt, rx_patt
        ).sel(polarization="y")

        # analytical value of rcs
        plate_area = (d_lam * lam0)**2
        rcs_expected = ((4 * np.pi) / lam0**2) * plate_area**2
        err = np.abs(conv.db10_lin(rcs_actual / rcs_expected))

        # theta = np.arange(-180, 182, 2)
        # rE_local_phi0 = antennas.get_scattering_rE(obj, Js, Ms, theta, np.ones_like(theta) * 0, frequency).squeeze()
        # rE_local_phi0.coords["phitheta"] = theta
        # rE_local_phi0 = rE_local_phi0.rename(phitheta="theta")

        # fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        # rE_local_phi0.plot("theta", xfmt=np.deg2rad, yfmt="db20", polarization="phipol")
        # plt.show()

        np.testing.assert_array_less(err, 0.01)



if __name__ == "__main__":
    unittest.main()