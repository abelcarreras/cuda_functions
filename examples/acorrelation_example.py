from cuda_functions import cuda_acorrelate

import numpy as np
import matplotlib.pyplot as pl
import cProfile, pstats


# Check function
if True:
    # Double precision
    data = np.sin(np.arange(0, 50, 0.1)) + np.cos(np.arange(0, 50, 0.1)) * 1.0j

    # Single precision
    # data = np.array(data, dtype='complex64')

    import matplotlib.pyplot as pl

    pl.plot(data, label='original data')

    cuda_res = cuda_acorrelate(data, mode="same") / data.size
    numpy_res = np.correlate(data, data, mode='same') / data.size

    pl.plot(cuda_res.real, '.', label='cuda real', marker='o')
    pl.plot(cuda_res.imag, '.', label='cuda imag', marker='o')
    pl.plot(numpy_res.real, label='numpy real')
    pl.plot(numpy_res.imag, label='numpy imag')

    pl.legend()
    pl.show()


# Test speed
if True:
    # Initialize GPU for a fair start
    signal = np.random.rand(100) + np.random.rand(100)*1j
    cProfile.run('cuda_acorrelate(signal, mode="same")')

    time_n = []
    time_ns = []
    time_c = []
    time_cs = []

    tlen = []
    for l in range(100, 200000, 50000):

        # Double precision
        signal = np.random.rand(l) + np.random.rand(l)*1j

        # Single precision
        signal_64 = np.array(signal, dtype='complex64')

        cProfile.run('np.correlate(signal, signal, mode="full")', 'restats_n')
        cProfile.run('np.correlate(signal_64, signal_64, mode="full")', 'restats_ns')
        cProfile.run('cuda_acorrelate(signal_64, mode="full", safe_mode=False)', 'restats_cs')
        cProfile.run('cuda_acorrelate(signal, mode="full", safe_mode=False)', 'restats_c')

        p_n = pstats.Stats('restats_n')
        p_ns = pstats.Stats('restats_ns')
        p_c = pstats.Stats('restats_c')
        p_cs = pstats.Stats('restats_cs')

        print ('{:8d} {:.6f} {:.6f} {:.6f} {:.6f}'.format(l,
                                                         p_n.total_tt / p_n.total_calls,
                                                         p_ns.total_tt / p_ns.total_calls,
                                                         p_c.total_tt / p_c.total_calls,
                                                         p_cs.total_tt / p_cs.total_calls))

        time_n.append(p_n.total_tt / p_n.total_calls)
        time_ns.append(p_ns.total_tt / p_ns.total_calls)
        time_c.append(p_c.total_tt / p_c.total_calls)
        time_cs.append(p_cs.total_tt / p_cs.total_calls)
        tlen.append(l)

    pl.plot(tlen, time_n, label='numpy double', marker='o')
    pl.plot(tlen, time_ns, label='numpy single', marker='o', color='y')
    pl.plot(tlen, time_c, label='cuda double', marker='o', color='r')
    pl.plot(tlen, time_cs, label='cuda single', marker='o', color='g')
    pl.xlabel('number of samples')
    pl.ylabel('time [s]')
    pl.legend()

    pl.show()

