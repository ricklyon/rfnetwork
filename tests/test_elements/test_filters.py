import unittest
import numpy as np
from pathlib import Path

import rfnetwork as rfn
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


DATA_DIR = Path(__file__).parent.parent / "data"


class TestCore(unittest.TestCase):

    def test_bandpass(self):
        bpf = rfn.elements.BandPassFilter(fc1=1e9, fc2=1.3e9, n=3)

        frequency = np.arange(10e6, 4e9, 10e6)

        ref_bpf = rfn.Component_SnP(DATA_DIR / "test_bpf.s2p")

        ref_data = ref_bpf.evaluate(frequency)["s"]
        test_data = bpf.evaluate(frequency)["s"]

        np.testing.assert_array_almost_equal(ref_data, test_data, decimal=1)

        # ax = plt.axes()
        # bpf.plot(ax, frequency, 21, freq_unit="mhz")
        # ax.set_ylim([-50, 0]);


    def test_bandstop(self):
        bsf = rfn.elements.BandStopFilter(fc1=1e9, fc2=1.3e9, n=3)

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

        lpf = rfn.elements.LowPassFilter(fc, n=5)

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

        hpf = rfn.elements.HighPassFilter(fc, n=5)

        ref_hpf = rfn.Component_SnP(DATA_DIR / "test_hpf.s2p")

        ref_data = ref_hpf.evaluate(frequency)["s"]
        test_data = hpf.evaluate(frequency)["s"]

        np.testing.assert_array_almost_equal(ref_data, test_data, decimal=2)

        # ax = plt.axes()
        # hpf.plot(ax, frequency, 21, freq_unit="mhz")
        # ax.set_ylim([-50, 0]);


if __name__ == "__main__":
    unittest.main()