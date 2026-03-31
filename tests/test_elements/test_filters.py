import unittest
import numpy as np
from pathlib import Path
from parameterized import parameterized
import itertools

import rfnetwork as rfn
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


DATA_DIR = Path(__file__).parent.parent / "data"


class TestFilters(unittest.TestCase):

    @parameterized.expand(
        list(range(1, 11))
    )
    def test_butterworth_prototypes(self, n):
        ref_table = {
            1: [2.0000, 1.0000],
            2: [1.4142, 1.4142, 1.0000],
            3: [1.0000, 2.0000, 1.0000, 1.0000],
            4: [0.7654, 1.8478, 1.8478, 0.7654, 1.0000],
            5: [0.6180, 1.6180, 2.0000, 1.6180, 0.6180, 1.0000],
            6: [0.5176, 1.4142, 1.9318, 1.9318, 1.4142, 0.5176, 1.0000],
            7: [0.4450, 1.2470, 1.8019, 2.0000, 1.8019, 1.2470, 0.4450, 1.0000],
            8: [0.3902, 1.1111, 1.6629, 1.9615, 1.9615, 1.6629, 1.1111, 0.3902, 1.0000],
            9: [0.3473, 1.0000, 1.5321, 1.8794, 2.0000, 1.8794, 1.5321, 1.0000, 0.3473, 1.0000],
            10: [0.3129, 0.9080, 1.4142, 1.7820, 1.9754, 1.9754, 1.7820, 1.4142, 0.9080, 0.3129, 1.0000]
        }[n]

        np.testing.assert_array_almost_equal(ref_table, rfn.utils.butterworth_prototype(n)[1:], decimal=3)

    @parameterized.expand(
        list(itertools.product(range(1, 11), (0.5, 3)))
    )
    def test_chebyshev_prototypes(self, n, ripple):

        """
        References
        ----------
        [1] Pozar, David M. Microwave Engineering. 4th ed., Wiley, 2011.

        """
        ref_table = {
            # 0.5dB Ripple.
            0.5 : {
                1: [0.6986, 1.0000],
                2: [1.4029, 0.7071, 1.9841],
                3: [1.5963, 1.0967, 1.5963, 1.0000],
                4: [1.6703, 1.1926, 2.3661, 0.8419, 1.9841],
                5: [1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0000],
                6: [1.7254, 1.2479, 2.6064, 1.3137, 2.4758, 0.8696, 1.9841],
                7: [1.7372, 1.2583, 2.6381, 1.3444, 2.6381, 1.2583, 1.7372, 1.0000],
                8: [1.7451, 1.2647, 2.6564, 1.3590, 2.6964, 1.3389, 2.5093, 0.8796, 1.9841],
                9: [1.7504, 1.2690, 2.6678, 1.3673, 2.7239, 1.3673, 2.6678, 1.2690, 1.7504, 1.0000],
                10: [1.7543, 1.2721, 2.6754, 1.3725, 2.7392, 1.3806, 2.7231, 1.3485, 2.5239, 0.8842, 1.9841]
            },
            # 3.0dB Ripple.
            3.0 : {
                1: [1.9953, 1.0000],
                2: [3.1013, 0.5339, 5.8095],
                3: [3.3487, 0.7117, 3.3487, 1.0000],
                4: [3.4389, 0.7483, 4.3471, 0.5920, 5.8095],
                5: [3.4817, 0.7618, 4.5381, 0.7618, 3.4817, 1.0000],
                6: [3.5045, 0.7685, 4.6061, 0.7929, 4.4641, 0.6033, 5.8095],
                7: [3.5182, 0.7723, 4.6386, 0.8039, 4.6386, 0.7723, 3.518, 1.0000],
                8: [3.5277, 0.7745, 4.6575, 0.8089, 4.6990, 0.8018, 4.499, 0.6073, 5.8095],
                9: [3.5340, 0.7760, 4.6692, 0.8118, 4.7272, 0.8118, 4.669, 0.7760, 3.5340, 1.0000],
                10: [3.5384, 0.7771, 4.6768, 0.8136, 4.7425, 0.816, 4.726, 0.8051, 4.5142, 0.6091, 5.8095]
            }
        }[ripple][n]

        np.testing.assert_array_almost_equal(ref_table, rfn.utils.chebyshev_prototype(n, ripple)[1:], decimal=3)


    def test_bandpass(self):
        bpf = rfn.elements.LumpedElementFilter.from_chebyshev(fc=(1e9, 1.3e9), btype="bandpass", n=3)

        frequency = np.arange(10e6, 4e9, 10e6)

        ref_bpf = rfn.Component_SnP(DATA_DIR / "test_bpf.s2p")

        ref_data = ref_bpf.evaluate(frequency)["s"]
        test_data = bpf.evaluate(frequency)["s"]

        np.testing.assert_array_almost_equal(ref_data, test_data, decimal=1)

        # ax = plt.axes()
        # bpf.plot(ax, frequency, 21, freq_unit="mhz")
        # ax.set_ylim([-50, 0]);


    def test_bandstop(self):
        bsf = rfn.elements.LumpedElementFilter.from_chebyshev(fc=(1e9, 1.3e9), btype="bandstop", n=3)

        frequency = np.arange(10e6, 3e9, 10e6)

        ref_bsf = rfn.Component_SnP(DATA_DIR / "test_bsf.s2p")

        ref_data = ref_bsf.evaluate(frequency)["s"]
        test_data = bsf.evaluate(frequency)["s"]

        np.testing.assert_array_almost_equal(ref_data, test_data, decimal=1)

        # ax = plt.axes()
        # bsf.plot(ax, frequency, 21, freq_unit="mhz")
        # ax.set_ylim([-50, 0]);


    def test_lowpass(self):
        fc = 1.5e9
        
        frequency = np.arange(10e6, 3e9, 10e6)

        lpf = rfn.elements.LumpedElementFilter.from_chebyshev(fc=fc, btype="lowpass", n=5)

        ref_lpf = rfn.Component_SnP(DATA_DIR / "test_lpf.s2p")

        ref_data = ref_lpf.evaluate(frequency)["s"]
        test_data = lpf.evaluate(frequency)["s"]

        np.testing.assert_array_almost_equal(ref_data, test_data, decimal=2)

        # ax = plt.axes()
        # lpf.plot(ax, frequency, 21, freq_unit="mhz")
        # ax.set_ylim([-50, 0]);

    def test_highpass(self):
        fc = 1.5e9
        
        frequency = np.arange(10e6, 3e9, 10e6)

        hpf = rfn.elements.LumpedElementFilter.from_chebyshev(fc=fc, btype="highpass", n=5)

        ref_hpf = rfn.Component_SnP(DATA_DIR / "test_hpf.s2p")

        ref_data = ref_hpf.evaluate(frequency)["s"]
        test_data = hpf.evaluate(frequency)["s"]

        np.testing.assert_array_almost_equal(ref_data, test_data, decimal=2)

        # ax = plt.axes()
        # hpf.plot(ax, frequency, 21, freq_unit="mhz")
        # ax.set_ylim([-50, 0]);


if __name__ == "__main__":
    unittest.main()