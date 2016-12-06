
import extensions.gpu_correlate as gpu_correlation
import extensions.gpu_fft as gpu_fft
import numpy as np
import matplotlib.pyplot as pl


data = np.sin(np.arange(0, 8000, 0.1)) + np.random.rand(80000)*0.1 # + np.cos(np.arange(0, 800, 0.1))*1.0j

#pl.plot(data, label='original')

cuda_res = (gpu_correlation.dacorrelate(data, mode="same")) / data.size
cuda_res = gpu_fft.dfft(cuda_res)

numpy_res = np.correlate(data, data, mode='same') / data.size
numpy_res = np.fft.fft(numpy_res)


freqs = np.fft.fftfreq(numpy_res.size, 0.01)
idx = np.argsort(freqs)

frequency_range = np.arange(0.8, 2, 0.01)

cuda_res = np.abs(cuda_res)
cuda_res =np.interp(frequency_range, freqs[idx], cuda_res[idx].real)

numpy_res = np.abs(numpy_res)
numpy_res =np.interp(frequency_range, freqs[idx], numpy_res[idx].real)



pl.plot(frequency_range, cuda_res, '.', label='cuda', marker='o')
#pl.plot(frequency_range, res2.imag, '.', label='cuda imag', marker='o')
pl.plot(frequency_range, numpy_res, label='numpy')
#pl.plot(frequency_range, res3.imag, label='numpy imag')

pl.legend()
pl.show()

