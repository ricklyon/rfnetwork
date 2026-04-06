import unittest
import numpy as np
from pathlib import Path
from np_struct import ldarray

import rfnetwork as rfn

DATA_DIR = Path(__file__).parent / "../data"


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

    def test_eng_formatter(self):

        self.assertEqual(
            rfn.utils.eng_formatter([10000000000, 0.02, 11e8]), '[10e9 0.02 1.1e9]'
        )

        self.assertEqual(
            rfn.utils.eng_formatter([0, 0.0223e-9]), '[0 22.3e-12]'
        )

        self.assertEqual(
            rfn.utils.eng_formatter(None), 'None'
        )

        self.assertEqual(
            rfn.utils.eng_formatter(10.005e-8), '100.05e-9'
        )

        self.assertEqual(
            rfn.utils.eng_formatter(5555555), '5.556e6'
        )

        self.assertEqual(
            rfn.utils.eng_formatter(2e7 +0.01j), '(20e6+0.01j)'
        )

        self.assertEqual(
            rfn.utils.eng_formatter([2e7 +0.01j, 2e-4j]), '[(20e6+0.01j) (0+200e-6j)]'
        )

    def test_gerber_import(self):

        img = rfn.utils_mesh.get_gerber_image(DATA_DIR / "lab_project-B_Cu.gbr", origin=(-2.0197, -0.0197), dpi=100)

        # compare with regression file
        img_ref = ldarray.load(DATA_DIR / "regression/lab_project-B_Cu.npy")

        np.testing.assert_array_almost_equal(img, img_ref)
        # check coordinates
        np.testing.assert_array_almost_equal(img.coords["x"], img_ref.coords["x"], decimal=4)
        np.testing.assert_array_almost_equal(img.coords["y"], img_ref.coords["y"], decimal=4)

if __name__ == "__main__":
    unittest.main()