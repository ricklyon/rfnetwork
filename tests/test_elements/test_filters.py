import unittest
import numpy as np

import rfnetwork as rfn
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)



class TestCore(unittest.TestCase):

    def test_bandpass(self):
        bpf = rfn.elements.BandPassFilter(fc1=1e9, fc2=1.3e9, n=3)

        bpf.state

        frequency = np.arange(10e6, 3e9, 10e6)

        ax = plt.axes()
        bpf.plot(ax, frequency, 21, freq_unit="mhz")
        ax.set_ylim([-50, 0]);

    def test_bandstop(self):
        bsf = rfn.elements.BandStopFilter(fc1=1e9, fc2=1.3e9, n=3)

        bsf.state

        frequency = np.arange(10e6, 3e9, 10e6)

        ax = plt.axes()
        bsf.plot(ax, frequency, 21, freq_unit="mhz")
        ax.set_ylim([-50, 0]);


    def test_lowpass(self):
        fc = 1.5e9
        
        frequency = np.arange(10e6, 3e9, 10e6)

        lpf = rfn.elements.LowPassFilter(fc, n=5)

        ax = plt.axes()
        lpf.plot(ax, frequency, 21, freq_unit="mhz")
        ax.set_ylim([-50, 0]);

    def test_highpass(self):
        fc = 1.5e9
        
        frequency = np.arange(10e6, 3e9, 10e6)

        lpf = rfn.elements.HighPassFilter(fc, n=5)

        ax = plt.axes()
        lpf.plot(ax, frequency, 21, freq_unit="mhz")
        ax.set_ylim([-50, 0]);


if __name__ == "__main__":
    unittest.main()