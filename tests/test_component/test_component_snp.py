import unittest
import numpy as np
from pathlib import Path

import rfnetwork as rfn
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"


class TestComponentSnP(unittest.TestCase):

    def test_evaluate_noise(self):

        comp = rfn.Component_SnP(DATA_DIR / "QPL9503_5V_with_NP.s2p")
        frequency = np.arange(1e9, 6e9, 10e6)
        data = comp.evaluate(frequency, noise=True)
        # check frequency coordinates
        np.testing.assert_array_almost_equal(data["s"].coords["frequency"], frequency)
        # regression test the correlation matrix and s-matrix
        ref_sdata = np.load(DATA_DIR / "regression/test_evaluate_noise_s.npy")
        ref_ndata = np.load(DATA_DIR / "regression/test_evaluate_noise_n.npy")

        np.testing.assert_almost_equal(ref_sdata, data["s"])
        np.testing.assert_almost_equal(ref_ndata, data["n"])

        # regression test noise figure
        nf = rfn.core.noise_figure_from_ndata(data["s"], data["n"], (2, 1))
        ref_nf = np.load(DATA_DIR / "regression/test_evaluate_noise_nf.npy")
        np.testing.assert_almost_equal(ref_nf, nf)

    def test_default_frequency(self):
        # test that evaluate will default to the frequency found in the snp file
        comp = rfn.Component_SnP(file=DATA_DIR / "TQP3M9038.s2p")
        sdata = comp.evaluate()["s"]

        np.testing.assert_array_almost_equal(sdata.coords["frequency"], np.arange(10e6, 6.005e9, 5e6))

    def test_set_state(self):
        # test switching a component state
        comp = rfn.Component_SnP(
            dict(rf1=DATA_DIR / "PE4257/D_PE4257_RF1.s3p", rf2=DATA_DIR / "PE4257/D_PE4257_RF2.s3p")
        )

        frequency = np.arange(0.5e9, 3e9, 100e6)
        # check that default state is the first item in the dictionary
        np.testing.assert_equal(comp.state["file"], "rf1")

        # check that port 2 is connected to port 1 for the RF1 state
        rf1_sdb = rfn.conv.db20_lin(comp.evaluate(frequency)["s"])
        # check insertion loss on port 2
        np.testing.assert_array_less(-rf1_sdb.sel(b=2, a=1), 1.1)
        # check port 3 is isolated
        np.testing.assert_array_less(rf1_sdb.sel(b=3, a=1), -40)
        
        # switch to RF2 and check that port 3 is connected to port 1
        comp.set_state(file="rf2")
        rf2_sdb = rfn.conv.db20_lin(comp.evaluate(frequency)["s"])
        # check insertion loss on port 3
        np.testing.assert_array_less(-rf2_sdb.sel(b=3, a=1), 1.1)
        # check port 2 is isolated
        np.testing.assert_array_less(rf2_sdb.sel(b=2, a=1), -40)
        

    def test_file_not_found(self):
        
        with self.assertRaises(ValueError, msg="File does not exist:"):
            rfn.Component_SnP(file=DATA_DIR / "not_found.s2p")


if __name__ == "__main__":
    unittest.main()