import unittest
import numpy as np
from pathlib import Path

import rfnetwork as rfn
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).parent.parent / "data"


class TestNoiseNetwork(unittest.TestCase):

    def test_amplifier(self):
        # check noise figure of a single amplifier against a reference from ADS
        lna = rfn.Component_SnP(file=DATA_DIR / "QPL9503_5V_with_NP.s2p")

        class single_amp(rfn.Network):
            u1 = lna
            att = rfn.elements.Attenuator(0.5)

            cascades = [
                ("P1", att, u1, "P2")
            ]

        amp_n = single_amp()

        # reference noise figure from ADS
        ads_nf = np.genfromtxt(DATA_DIR / 'ads_lna_nf.csv', delimiter=",")[:, 1]

        # compute noise figure of network
        frequency = np.arange(0.6, 6.01, 0.01) * 1e9
        test_data = amp_n.evaluate(frequency, noise=True)
        # convert noise correlation matrix to noise figure in dB
        test_nf = rfn.conv.db10_lin(rfn.core.noise_figure_from_ndata(test_data["s"], test_data["n"], (2, 1)))

        np.testing.assert_array_almost_equal(test_nf, ads_nf, decimal=1)

    def test_wilk_network(self):
        # build a 3-port network of amplifiers and check noise figure against ADS reference

        line50 = rfn.elements.Line(z0=50)
        line70p7 = rfn.elements.Line(z0=70.7)
        f0 = 3e9

        class single_amp(rfn.Network):
            u1 = rfn.Component_SnP(file=DATA_DIR / "QPL9503_5V_with_NP.s2p")
            att = rfn.elements.Attenuator(0.5)

            cascades = [
                ("P1", att, u1, "P2")
            ]

        class wilk(rfn.Network):
            """
            Isolated wilkison combiner/splitter
            """
            p1 = line50(20, f0=f0)
            p2 = line70p7(90, f0=f0)
            p3 = line70p7(90, f0=f0)
            r1 = rfn.elements.Resistor(100)

            nodes = [
                (p1|2, p2|1, p3|1),
                (p2|2, r1|1, "P2"),
                (p3|2, r1|2, "P3"),
                (p1|1, "P1"),
            ]


        class dual_amp(rfn.Network):
            u1 = single_amp()
            u2 = single_amp()
            sp = wilk()

            nodes = [
                (sp|1, "P1"),
                (sp|2, u1|1),
                (sp|3, u2|1),
                (u1|2, "P2"),
                (u2|2, "P3"),
            ]


        amp_dual_n = dual_amp()

        # compute noise figure of network
        frequency = np.arange(0.6, 6.01, 0.01) * 1e9
        test_data = amp_dual_n.evaluate(frequency, noise=True)
        # convert noise correlation matrix to noise figure in dB
        test_nf = rfn.conv.db10_lin(rfn.core.noise_figure_from_ndata(test_data["s"], test_data["n"], (2, 1)))

        # reference noise figure from ADS
        ads_nf = np.genfromtxt(DATA_DIR / 'ads_wilk_nf.csv', delimiter=",")[:, 1]

        # allow 0.5 of deviation from ADS reference
        diff_nf = np.abs(ads_nf - test_nf)
        np.testing.assert_array_less(diff_nf, 0.5)
        np.testing.assert_array_less(np.mean(diff_nf), 0.15)



if __name__ == "__main__":
    unittest.main()