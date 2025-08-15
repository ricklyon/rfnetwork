import unittest
import numpy as np
from pathlib import Path

import rfnetwork as rfn
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

from rfnetwork.core import core_func

DATA_DIR = Path(__file__).parent.parent / "data"

class TestConnectSelf(unittest.TestCase):

    def test_connect_self_wprobes(self):
        s1_comp = rfn.Component_SnP(DATA_DIR / "PE4257/D_PE4257_RF1.s3p")

        frequency = np.arange(2e9, 2.9e9, 100e6)

        s1 = s1_comp.evaluate(frequency)["s"]
        c1 = s1_comp.evaluate(frequency, noise=True)["n"]

        connections = [(2, 1)]
        probes = [(1, 1)]
                
        ref_data = np.load(DATA_DIR / "regression/test_connect_self.npz")
        cas_s_ref = ref_data["s"]
        cas_n_ref = ref_data["n"]
        row_order_ref = ref_data["row_order"]

        connections = np.array(np.atleast_2d(connections), dtype=np.int32, order="C")
        probes = np.array(np.atleast_2d(probes), dtype=np.int32, order="C")
        n_connections = len(connections)

        s1_b, s1_a = s1.shape[-2:]
        a_len = s1_a - (2 * n_connections)

        n_row = s1_b - (2 * n_connections) + np.count_nonzero(probes)
        row_order = np.zeros(n_row, dtype=np.int32, order="C")

        cas_s = np.zeros((len(s1), len(row_order), a_len), dtype="complex128")
        cas_n = np.zeros((len(s1), a_len, a_len), dtype="complex128")

        row_order = np.zeros_like(row_order, dtype=np.int32, order="C")
        core_func.connect_self(s1, c1, connections, probes, row_order, cas_s, cas_n)

        np.testing.assert_array_almost_equal(cas_s_ref, cas_s)
        np.testing.assert_array_almost_equal(cas_n_ref / (rfn.const.k), cas_n / (rfn.const.k))
        np.testing.assert_array_almost_equal(row_order_ref, row_order)

        # row_order_py, cas_py = core.connect_self_py(
        #   s1_comp.evaluate(frequency, noise=True), *connections[0], probes[0], noise=True
        # )
        # np.testing.assert_array_almost_equal(cas_py["s"], cas_s)
        # np.testing.assert_array_almost_equal(cas_py["n"] / (rfn.const.k), cas_n / (rfn.const.k))
        # np.savez(DATA_DIR / "regression/test_connect_self.npz", s=cas_s, n=cas_n, row_order=row_order)


if __name__ == "__main__":
    unittest.main()