from cuda_functions.bin import gpu_correlate_dp, gpu_correlate_sp, gpu_correlate_spc, gpu_correlate_dpc
from cuda_functions.bin import gpu_fft_dpc, gpu_fft_spc

import numpy as np
import warnings

def cuda_acorrelate(array, mode='valid', safe_mode=True):
    if safe_mode:
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_correlate_spc.acorrelate(array, mode=mode)
    elif array.dtype == 'complex128':
        return gpu_correlate_dpc.acorrelate(array, mode=mode)
    elif array.dtype == 'float32':
        return gpu_correlate_sp.acorrelate(array, mode=mode)
    elif array.dtype == 'float64':
        return gpu_correlate_dp.acorrelate(array, mode=mode)
    else:
        warnings.warn('{} type not supported, it will be converted to complex128'.format(array.dtype))
        return gpu_correlate_dpc.acorrelate(np.array(array, dtype='complex128'), mode=mode)


def cuda_fft(array, safe_mode=True):
    if safe_mode:
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_fft_spc.fft(array)
    elif array.dtype == 'complex128':
        return gpu_fft_dpc.fft(array)
    else:
        warnings.warn('{} type not supported, it will be converted to complex128'.format(array.dtype))
        return gpu_fft_dpc.fft(np.array(array, dtype='complex128'))


def cuda_ifft(array, safe_mode=True):
    if safe_mode:
        array = np.ascontiguousarray(array)

    if array.dtype == 'complex64':
        return gpu_fft_spc.ifft(array)
    elif array.dtype == 'complex128':
        return gpu_fft_dpc.ifft(array)
    else:
        warnings.warn('{} type not supported, it will be converted to complex128'.format(array.dtype))
        return gpu_fft_dpc.ifft(np.array(array, dtype='complex128'))