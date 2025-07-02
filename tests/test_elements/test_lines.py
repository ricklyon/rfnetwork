import unittest
import numpy as np
from pathlib import Path

import rfnetwork as rfn
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"



class TestCore(unittest.TestCase):

    def test_microstrip_delay(self):
        # use line parameters for 2x2116 (10mil) FR-4 core.
        # https://www.isola-group.com/pcb-laminates-prepreg/fr408hr-laminate-and-prepreg/

        length = 3.5
        msline = rfn.elements.MSLine(
            h=0.01,
            er=[3.72, 3.71, 3.70, 3.69, 3.67, 3.67],
            w=0.02,
            df=[0.0070, 0.0077, 0.0086, 0.0089, 0.0095, 0.0093],
            frequency=[100e6, 500e6, 1e9, 2e9, 5e9, 10e9],
            length=3.5
        )

        frequency = np.arange(0.01, 10, 0.01) * 1e9
        wavelength = msline.get_wavelength(frequency)

        er_eff = msline.get_properties(frequency).sel(value="er")
        vp = rfn.const.c0_in / np.sqrt(er_eff)
        ref_wavelength = vp / frequency

        np.testing.assert_almost_equal(wavelength, ref_wavelength)

        phs, td = msline.get_delay(frequency)

        np.testing.assert_almost_equal((length / vp) * 1e12, td)
        np.testing.assert_almost_equal(np.rad2deg((2 * np.pi * length) / wavelength), phs)


    def test_microstrip_fr4(self):

        msline = rfn.elements.MSLine(
            h=0.01,
            er=3.7,
            w=0.02,
            df=0.0086,
            length=3.5
        )

        # reference is from a AWR sim of a MLIN component with the same properties
        ref_msline = rfn.Component_SnP(DATA_DIR / "test_msline.s2p")

        frequency = np.arange(0.01, 10, 0.01) * 1e9

        test_data = msline.evaluate(frequency)["s"]
        ref_data = ref_msline.evaluate(frequency)["s"]

        # coarse check
        np.testing.assert_array_almost_equal(test_data, ref_data, decimal=1)

        diff = test_data / ref_data

        # error of ang(S21) less than 5 degrees
        np.testing.assert_array_less(np.abs(np.angle(diff.sel(b=2, a=1), deg=True)), 5)

        # error in S21 magnitude less than 0.5dB and more lossy than reference
        np.testing.assert_array_less(np.abs(rfn.conv.db20_lin(diff.sel(b=2, a=1))), 0.5)
        np.testing.assert_array_less(rfn.conv.db20_lin(diff.sel(b=2, a=1)), 0)

        # check S12 and S21 are equal
        np.testing.assert_array_almost_equal(test_data.sel(b=1, a=2), test_data.sel(b=2, a=1))

        # check S22 and S11
        np.testing.assert_array_less(rfn.conv.db20_lin(test_data.sel(b=1, a=1)), -35)
        np.testing.assert_array_less(rfn.conv.db20_lin(test_data.sel(b=2, a=2)), -35)

        # ax = plt.axes()
        # msline.plot(ax, frequency, 11, fmt="db")
        # ref_msline.plot(ax, frequency, 11, fmt="db")

    def test_stripline(self):

        sline = rfn.elements.Stripline(
           w=0.008,
           b=0.010,
           er=3.0,
           df=0.05,
           length=0.5
        )

        # reference is from a AWR sim of a MLIN component with the same properties
        ref_sline = rfn.Component_SnP(DATA_DIR / "test_stripline.s2p")

        frequency = np.arange(100, 1e4, 10) * 1e6

        ax = plt.axes()
        sline.plot(ax, frequency, 21, fmt="db")
        ref_sline.plot(ax, frequency, 21, fmt="db")

        test_data = sline.evaluate(frequency)["s"]
        ref_data = ref_sline.evaluate(frequency)["s"]

        diff = test_data / ref_data

        # error of ang(S21) less than 5 degrees
        np.testing.assert_array_less(np.abs(np.angle(diff.sel(b=2, a=1), deg=True)), 2)

if __name__ == "__main__":
    unittest.main()