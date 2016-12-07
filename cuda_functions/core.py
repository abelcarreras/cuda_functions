from bin import gpu_correlate
from bin import gpu_fft

import numpy as np


def cuda_acorrelate(array, mode='valid', safe_mode=True):
    if safe_mode:
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_correlate.acorrelate(array, mode=mode)
    elif array.dtype == 'complex128':
        return gpu_correlate.dacorrelate(array, mode=mode)
    else:
        raise TypeError('{} type not supported (complex only)'.format(array.dtype))


def cuda_fft(array, safe_mode=True):
    if safe_mode:
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_fft.fft(array)
    elif array.dtype == 'complex128':
        return gpu_fft.dfft(array)
    else:
        raise TypeError('{} type not supported (complex only)'.format(array.dtype))


def cuda_ifft(array, safe_mode=True):
    if safe_mode:
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_fft.ifft(array)
    elif array.dtype == 'complex128':
        return gpu_fft.difft(array)
    else:
        raise TypeError('{} type not supported (complex only)'.format(array.dtype))