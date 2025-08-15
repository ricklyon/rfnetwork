import unittest
import numpy as np
from pathlib import Path

import rfnetwork as rfn
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

from rfnetwork.core import core_func

DATA_DIR = Path(__file__).parent.parent / "data"

class TestConnect(unittest.TestCase):

    def test_connect_wprobes(self):

        s1_comp = rfn.Component_SnP(DATA_DIR / "PE4257/D_PE4257_RF1.s3p")
        s2_comp = rfn.elements.Line(z0=50, loss=2, er=3, length=3)

        frequency = np.arange(2e9, 2.9e9, 100e6)

        s1 = s1_comp.evaluate(frequency, noise=True)
        s2 = s2_comp.evaluate(frequency, noise=True)

        connections = [(2, 1)]
        probes = [(1, 1)]

        ref_data = np.load(DATA_DIR / "regression/test_connect.npz")
        cas_s_ref = ref_data["s"]
        cas_n_ref = ref_data["n"]
        row_order_ref = ref_data["row_order"]
                        
        connections = np.array(np.atleast_2d(connections), dtype=np.int32, order="C")
        probes = np.array(np.atleast_2d(probes), dtype=np.int32, order="C")

        n_connections = len(connections)

        f_len, s1_b, s1_a = s1["s"].shape
        f_len, s2_b, s2_a = s2["s"].shape

        a_len = s1_a + s2_a - (2 * n_connections)
        b_len = s1_b + s2_b
        n_row = b_len - (2 * n_connections) + np.count_nonzero(probes)

        row_order = np.zeros(n_row, dtype=np.int32, order="C")

        cas_s = np.zeros((f_len, n_row, a_len), dtype="complex128")
        cas_n = np.zeros((f_len, a_len, a_len), dtype="complex128")

        core_func.connect_other(s1["s"], s2["s"], s1["n"], s2["n"], connections, probes, row_order, cas_s, cas_n)

        np.testing.assert_array_almost_equal(cas_s_ref, cas_s)
        np.testing.assert_array_almost_equal(cas_n_ref / (rfn.const.k), cas_n / (rfn.const.k))
        np.testing.assert_array_almost_equal(row_order_ref, row_order)

        # row_order_py, cas_py = core.connect_py(s1, s2, connections, probes, noise=True)
        # np.testing.assert_array_almost_equal(cas_py["s"], cas_s)
        # np.testing.assert_array_almost_equal(cas_py["n"] / (rfn.const.k), cas_n / (rfn.const.k))
        # np.testing.assert_array_almost_equal(row_order, row_order_py)
        # np.savez(DATA_DIR / "regression/test_connect.npz", s=cas_s, n=cas_n, row_order=row_order)

if __name__ == "__main__":
    unittest.main()