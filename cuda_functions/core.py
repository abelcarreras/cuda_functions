from cuda_functions.bin import gpu_correlate
from cuda_functions.bin import gpu_fft

import numpy as np
import warnings

def cuda_acorrelate(array, mode='valid', safe_mode=True):
    if safe_mode:
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_correlate.acorrelate(array, mode=mode)
    elif array.dtype == 'complex128':
        return gpu_correlate.dacorrelate(array, mode=mode)
    else:
        warnings.warn('{} type not supported, it will be converted to complex128'.format(array.dtype))
        return gpu_correlate.dacorrelate(np.array(array, dtype='complex128'), mode=mode)


def cuda_fft(array, safe_mode=True):
    if safe_mode:
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_fft.fft(array)
    elif array.dtype == 'complex128':
        return gpu_fft.dfft(array)
    else:
        warnings.warn('{} type not supported, it will be converted to complex128'.format(array.dtype))
        return gpu_fft.dfft(np.array(array, dtype='complex128'))


def cuda_ifft(array, safe_mode=True):
    if safe_mode:
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_fft.ifft(array)
    elif array.dtype == 'complex128':
        return gpu_fft.difft(array)
    else:
        warnings.warn('{} type not supported, it will be converted to complex128'.format(array.dtype))
        return gpu_fft.difft(np.array(array, dtype='complex128'))