import unittest
import numpy as np

from rfnetwork.core import core_func
from time import time
from parameterized import parameterized


class TestCascadeNoise(unittest.TestCase):

    @parameterized.expand([[1, 3], [3, 1], [2, 2], [5, 3], [3, 5]])
    def test_cascade_noise_data(self, m1_a, m2_a):

        flen = 20
        m_b = m1_a + m2_a
        m1 = np.random.uniform(0, 1, size=(flen, m_b, m1_a)) + 1j * np.random.uniform(0, 1, size=(flen, m_b, m1_a))
        m2 = np.random.uniform(0, 1, size=(flen, m_b, m2_a)).astype("complex128")

        c1 = np.random.uniform(0, 1, size=(flen, m1_a, m1_a)) + 1j * np.random.uniform(0, 1, size=(flen, m1_a, m1_a))
        c2 = np.random.uniform(0, 1, size=(flen, m2_a, m2_a)).astype("complex128")

        out = np.zeros((flen, m_b, m_b), dtype="complex128")

        stime = time()
        core_func.cascade_ndata(m1, m2, c1, c2, out)
        elapsed = time() - stime

        out_np = np.zeros((flen, m_b, m_b), dtype="complex128")

        for f in range(flen):
            for i in range(m_b):
                for j in range(m_b):
                    out_np[f, i, j] = np.sum(np.outer(m1[f,i], m1[f,j].conj()) * c1[f])
                    out_np[f, i, j] += np.sum(np.outer(m2[f,i], m2[f,j].conj()) * c2[f])

        np.testing.assert_array_almost_equal(out, out_np)

    @parameterized.expand([1, 2, 3, 4, 5])
    def test_cascade_self_noise_data(self, m1_a):

        flen = 10
        m_b = m1_a
        m1 = np.random.uniform(0, 1, size=(flen, m_b, m_b)) + 1j * np.random.uniform(0, 1, size=(flen, m_b, m_b))
        c1 = np.random.uniform(0, 1, size=(flen, m_b, m_b)) + 1j * np.random.uniform(0, 1, size=(flen, m_b, m_b))

        out = np.zeros((flen, m_b, m_b), dtype="complex128")

        stime = time()
        core_func.cascade_self_ndata(m1, c1, out)
        elapsed = time() - stime

        out_np = np.zeros((flen, m_b, m_b), dtype="complex128")

        for f in range(flen):
            for i in range(m_b):
                for j in range(m_b):
                    out_np[f, i, j] = np.sum(np.outer(m1[f,i], m1[f,j].conj()) * c1[f])

        np.testing.assert_array_almost_equal(out, out_np)

if __name__ == "__main__":
    unittest.main()