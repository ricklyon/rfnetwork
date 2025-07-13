import unittest
import numpy as np
from pathlib import Path

import rfnetwork as rfn
import matplotlib.pyplot as plt
from parameterized import parameterized

DATA_DIR = Path(__file__).parent.parent / "data"


class TestAttenuator(unittest.TestCase):

    @parameterized.expand((0.5, 3, 10, -1, -3, -10))
    def test_pi_atten(self, attenuation):
        
        frequency = np.arange(10, 10e9, 10e6)
        attn = rfn.elements.PiAttenuator(attenuation)
        sdata = attn.evaluate(frequency)["s"]

        # check loss, S12, S21
        np.testing.assert_array_almost_equal(sdata.sel(b=2, a=1), rfn.conv.lin_db20(-np.abs(attenuation)))
        np.testing.assert_array_almost_equal(sdata.sel(b=1, a=2), rfn.conv.lin_db20(-np.abs(attenuation)))

        # check S11/S22
        np.testing.assert_array_almost_equal(sdata.sel(b=1, a=1), 0)
        np.testing.assert_array_almost_equal(sdata.sel(b=2, a=2), 0)

    @parameterized.expand((0.5, 3, 10, -1, -3, -10))
    def test_atten(self, attenuation):
        
        frequency = np.arange(10, 10e9, 10e6)
        attn = rfn.elements.Attenuator(attenuation)
        sdata = attn.evaluate(frequency)["s"]

        # check loss, S12, S21
        np.testing.assert_array_almost_equal(sdata.sel(b=2, a=1), rfn.conv.lin_db20(-np.abs(attenuation)))
        np.testing.assert_array_almost_equal(sdata.sel(b=1, a=2), rfn.conv.lin_db20(-np.abs(attenuation)))

        # check S11/S22
        np.testing.assert_array_almost_equal(sdata.sel(b=1, a=1), 0)
        np.testing.assert_array_almost_equal(sdata.sel(b=2, a=2), 0)



if __name__ == "__main__":
    unittest.main()