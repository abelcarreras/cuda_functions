from src import gpu_correlate
from src import gpu_fft

import numpy as np

def cuda_acorrelate(array, mode='valid', direct_interface=False):
    if not(direct_interface):
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_correlate.acorrelate(array, mode=mode)
    elif array.dtype == 'complex128':
        return gpu_correlate.dacorrelate(array, mode=mode)
    else:
        raise TypeError('{} type not supported (complex only)'.format(array.dtype))


def cuda_fft(array, direct_interface=False):
    if not(direct_interface):
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_fft.fft(array)
    elif array.dtype == 'complex128':
        return gpu_fft.dfft(array)
    else:
        raise TypeError('{} type not supported (complex only)'.format(array.dtype))


def cuda_ifft(array, direct_interface=False):
    if not(direct_interface):
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_fft.ifft(array)
    elif array.dtype == 'complex128':
        return gpu_fft.difft(array)
    else:
        raise TypeError('{} type not supported (complex only)'.format(array.dtype))