#!/usr/bin/env python
from cuda_functions import cuda_acorrelate
from cuda_functions import cuda_fft, cuda_ifft

import numpy as np
import unittest


class TestCuda(unittest.TestCase):

    def setUp(self):
        self.data = (np.sin(np.arange(0, 8000, 0.1)) * np.random.rand(80000) * 0.1 +
                     1.0j * np.cos(np.arange(0, 8000, 0.1)) * np.random.rand(80000) * 0.1)

    def test_acorrelation_float32(self):

        data = np.array(self.data.real, dtype='float32')

        cuda_res = cuda_acorrelate(data, mode="valid") / data.size
        # cuda_res = cuda_fft(np.array(cuda_res, dtype='complex64'))

        numpy_res = np.correlate(data, data, mode='valid') / data.size
        # numpy_res = np.fft.fft(np.array(numpy_res, dtype='complex64'))

        self.assertEqual(np.allclose(cuda_res, numpy_res, rtol=1, atol=1.e-8), True)

    def test_ps_float64(self):

        data = np.array(self.data.real, dtype='float64')

        cuda_res = cuda_acorrelate(data, mode='full') / data.size
        cuda_res = cuda_fft(np.array(cuda_res, dtype='complex128'))

        numpy_res = np.correlate(data, data, mode='full') / data.size
        numpy_res = np.fft.fft(numpy_res)

        self.assertEqual(np.allclose(cuda_res, numpy_res, rtol=1, atol=1.e-16), True)

    def test_ps_complex64(self):

        data = np.array(self.data, dtype='complex64')

        cuda_res = cuda_acorrelate(data, mode="same") / data.size
        cuda_res = cuda_fft(cuda_res)

        numpy_res = np.correlate(data, data, mode='same') / data.size
        numpy_res = np.fft.fft(numpy_res)


        self.assertEqual(np.allclose(cuda_res, numpy_res, rtol=1, atol=1.e-8), True)

    def test_ps_complex128(self):

        data = np.array(self.data, dtype='complex128')

        cuda_res = cuda_acorrelate(data, mode="full") / data.size
        cuda_res = cuda_fft(cuda_res)

        numpy_res = np.correlate(data, data, mode='full') / data.size
        numpy_res = np.fft.fft(numpy_res)

        self.assertEqual(np.allclose(cuda_res, numpy_res, rtol=1, atol=1.e-16), True)

    def test_fft_complex64(self):

        data = np.array(self.data, dtype='complex64')
        res_fft = cuda_fft(data)
        res_ifft = cuda_ifft(res_fft)

        self.assertEqual(np.allclose(data, res_ifft, rtol=1, atol=1.e-8), True)

    def test_fft_complex128(self):

        data = np.array(self.data, dtype='complex128')
        res_fft = cuda_fft(data)
        res_ifft = cuda_ifft(res_fft)

        self.assertEqual(np.allclose(data, res_ifft, rtol=1, atol=1.e-16), True)


if __name__ == '__main__':
    unittest.main()
