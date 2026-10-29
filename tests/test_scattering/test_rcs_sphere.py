import pyvista as pv
import unittest
from parameterized import parameterized

from np_struct import ldarray
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rfnetwork as rfn

from rfnetwork import const, conv, antennas
from scipy.special import spherical_jn, spherical_yn

def _mie_coeffs(x):
    n_max = int(x + 4 * x ** (1 / 3) + 4)
    n = np.arange(1, n_max + 1)
    jn, yn = spherical_jn(n, x), spherical_yn(n, x)
    jnp, ynp = spherical_jn(n, x, True), spherical_yn(n, x, True)
    hn, hnp = jn + 1j * yn, jnp + 1j * ynp
    # PEC limit of Bohren & Huffman coefficients; psi = x j_n, xi = x h_n
    a = (jn + x * jnp) / (hn + x * hnp)
    b = jn / hn
    return n, a, b

def mie_monostatic(ka, radius):
    n, a, b = _mie_coeffs(ka)
    s = np.sum((-1) ** n * (2 * n + 1) * (a - b))
    return np.pi * radius ** 2 * np.abs(s) ** 2 / ka ** 2


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

        ref_rcs = mie_monostatic(ka, r_lam * lam0)
        err = np.abs(conv.db10_lin(rcs_actual / ref_rcs))

        # check RCS is correct up to half a dB
        np.testing.assert_array_less(err, 0.5)

if __name__ == "__main__":
    unittest.main()