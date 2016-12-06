import extensions.gpu_fft as gpu_fft
import numpy as np
import matplotlib.pyplot as pl

import cProfile, pstats

# Check function
if True:
    signal = np.sin(np.arange(300) * 0.1) + np.cos(np.arange(300) * 0.1) * 1j

    pl.plot(signal.real)
    pl.plot(signal.imag)
    pl.show()

    print('numpy')
    a = np.fft.fft(signal)
    a = np.fft.ifft(a)
    print('finish numpy')

    print('cuda')
    b = gpu_fft.dfft(signal)
    b = gpu_fft.difft(b)
    print('finish cuda')

    # print(a)
    pl.plot(a.real, label='numpy real')
    pl.plot(a.imag, label='numpy imag')

    pl.plot(b.real, '.', label='CUDA real', marker='o')
    pl.plot(b.imag, '.', label='CUDA imag', marker='o')

    pl.legend()
    pl.show()

# Test speed
if False:
    # Initialize GPU for a fair start
    signal = np.random.rand(100) + np.random.rand(100)*1j
    cProfile.run('test.dfft(signal)')

    time_n = []
    time_c = []
    tlen = []
    for l in range(100000, 10000000, 100000):
        #signal = np.sin(np.arange(3000000)*0.1) + np.cos(np.arange(3000000)*0.1)*1j
        signal = np.random.rand(l) + np.random.rand(l)*1j


        signal = np.array(signal, dtype=np.complex64)

        cProfile.run('np.fft.fft(signal)', 'restats_n')
        cProfile.run('test.fft(signal)', 'restats_c')


        p_n = pstats.Stats('restats_n')
        p_c = pstats.Stats('restats_c')

        print '{} {} {}'.format(l,
                                p_n.total_tt/ p_n.total_calls,
                                p_c.total_tt/ p_c.total_calls)

        time_n.append(p_n.total_tt/p_n.total_calls)
        time_c.append(p_c.total_tt/p_c.total_calls)
        tlen.append(l)

    pl.plot(tlen, time_n, label='numpy', marker='o')
    pl.plot(tlen, time_c, label='cuda', marker='o', color='r')
    pl.legend()

    pl.show()


