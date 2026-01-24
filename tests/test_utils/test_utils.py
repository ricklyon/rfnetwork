import unittest
import numpy as np
from pathlib import Path

import rfnetwork as rfn
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).parent.parent / "data"


class TestUtils(unittest.TestCase):
    
    def test_combline_section_nb(self):

        # check narrowband example, page 619
        f0 = 1.5e9
        w = 0.1 * f0
        f1 = f0 - w/2
        f2 = f0 + w/2
        er = 1
        g_nb = [1, 1.1681, 1.4039, 2.0562, 1.5170, 1.9029, 0.8618, 1.3554]

        Ck, Cmk = rfn.utils.combline_sections_nb(g_nb, f1, f2, er, h=0.05143)

        # table 10.06-2
        np.testing.assert_array_almost_equal(
            Ck, [5.95, 3.39, 4.42, 4.496, 4.496, 4.420, 3.39, 5.95], decimal=2
        )
        # table 10.06-2
        np.testing.assert_array_almost_equal(
            Cmk, [1.582, 0.301, 0.226, 0.218, 0.226, 0.301, 1.582], decimal=2
        )

    def test_combline_section_wb(self):
        # check wideband example, page 632
        f0 = 1.5e9
        w = 0.7 * f0
        f1 = f0 - w/2
        f2 = f0 + w/2
        er = 1
        g_wb = [1, 1.1897, 1.4346, 2.1199, 1.6010, 2.1699, 1.5640, 1.9444, 0.8778, 1.3554]

        Ck, Cmk = rfn.utils.combline_sections_wb(g_wb, f1, f2, er, h=0.18)

        # table 10.06-2
        np.testing.assert_array_almost_equal(
            Ck, [2.235, 1.463, 1.675, 1.706, 1.706, 1.675, 1.463, 2.235], decimal=2
        )
        # table 10.06-2
        np.testing.assert_array_almost_equal(
            Cmk, [1.647, 1.115, 1.056, 1.044, 1.056, 1.115, 1.647], decimal=2
        )

if __name__ == "__main__":
    unittest.main()