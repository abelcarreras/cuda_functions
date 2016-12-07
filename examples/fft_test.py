from cuda_functions import cuda_fft, cuda_ifft
import numpy as np
import matplotlib.pyplot as pl

import cProfile, pstats

# Check function
if True:
    signal = np.sin(np.arange(300) * 0.1) + np.cos(np.arange(300) * 0.1) * 1j

    pl.plot(signal.real)
    pl.plot(signal.imag)
    pl.show()

    a = np.fft.fft(signal)
    a = np.fft.ifft(a)

    b = cuda_fft(signal)
    b = cuda_ifft(b)

    # print(a)
    pl.plot(a.real, label='numpy real')
    pl.plot(a.imag, label='numpy imag')

    pl.plot(b.real, '.', label='CUDA real', marker='o')
    pl.plot(b.imag, '.', label='CUDA imag', marker='o')

    pl.legend()
    pl.show()

# Test speed
if True:
    # Initialize GPU for a fair start
    signal = np.random.rand(100) + np.random.rand(100)*1j
    cProfile.run('cuda_fft(signal)')

    time_n = []
    time_ns = []
    time_c = []
    time_cs = []
    tlen = []
    for l in range(100000, 10000000, 1000000):
        #signal = np.sin(np.arange(3000000)*0.1) + np.cos(np.arange(3000000)*0.1)*1j
        signal = np.random.rand(l) + np.random.rand(l)*1j


        signal_64 = np.array(signal, dtype=np.complex64)

        cProfile.run('np.fft.fft(signal_64)', 'restats_ns')
        cProfile.run('np.fft.fft(signal)', 'restats_n')
        cProfile.run('cuda_fft(signal, safe_mode=False)', 'restats_c')
        cProfile.run('cuda_fft(signal_64, safe_mode=False)', 'restats_cs')


        p_n = pstats.Stats('restats_n')
        p_ns = pstats.Stats('restats_ns')
        p_c = pstats.Stats('restats_c')
        p_cs = pstats.Stats('restats_cs')

        print '{:8d} {:.6f} {:.6f} {:.6f} {:.6f}'.format(l,
                                p_n.total_tt / p_n.total_calls,
                                p_ns.total_tt/ p_ns.total_calls,
                                p_c.total_tt/ p_c.total_calls,
                                p_cs.total_tt / p_cs.total_calls)

        time_n.append(p_n.total_tt/p_n.total_calls)
        time_ns.append(p_ns.total_tt/p_ns.total_calls)
        time_c.append(p_c.total_tt/p_c.total_calls)
        time_cs.append(p_cs.total_tt/p_cs.total_calls)
        tlen.append(l)

    pl.plot(tlen, time_n, label='numpy double', marker='o')
    pl.plot(tlen, time_ns, label='numpy single', marker='o', color='y')
    pl.plot(tlen, time_c, label='cuda double', marker='o', color='r')
    pl.plot(tlen, time_cs, label='cuda single', marker='o', color='g')
    pl.xlabel('number of samples')
    pl.ylabel('time [s]')
    pl.legend()

    pl.show()


