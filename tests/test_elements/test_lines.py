import unittest
import numpy as np

import rfnetwork as rfn
import matplotlib.pyplot as plt



class TestCore(unittest.TestCase):

    def test_microstrip_fr4(self):
        # use line parameters for 2x2116 (10mil) FR-4 core.
        # https://www.isola-group.com/pcb-laminates-prepreg/fr408hr-laminate-and-prepreg/

        msline = rfn.elements.MSLine(
            h=0.01,
            er=[3.72, 3.71, 3.70, 3.69, 3.67, 3.67],
            w=0.02,
            loss_tan=[0.0070, 0.0077, 0.0086, 0.0089, 0.0095, 0.0093],
            frequency=[100e6, 500e6, 1e9, 2e9, 5e9, 10e9],
            length=3.5
        )

        msline()
        frequency = np.arange(100, 1010, 10) * 1e6

        # msline.set_state(w=0.05)
        msline.evaluate(frequency)

        ax = plt.axes()
        msline.plot(ax, frequency, 11, fmt="realz")


    def test_stripline(self):
        # use line parameters for 2x2116 (10mil) FR-4 core.
        # https://www.isola-group.com/pcb-laminates-prepreg/fr408hr-laminate-and-prepreg/

        sl = rfn.elements.Stripline(
           w=0.008,
           b=0.010,
           er=3.0,
           df=0.05,
           length=0.5
        )

        frequency = np.arange(100, 1e4, 10) * 1e6
        ax = plt.axes()
        sl.plot(ax, frequency, 11, fmt="db")


if __name__ == "__main__":
    unittest.main()