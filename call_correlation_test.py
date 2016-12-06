import functions.gpu_correlate as gpu_correlate
import numpy as np
import matplotlib.pyplot as pl

import cProfile, pstats

# Check function
if True:
    data = np.sin(np.arange(0, 500, 0.1)) + np.cos(np.arange(0, 500, 0.1)) * 1.0j

  #  data = np.array(data, dtype='complex64')
    import matplotlib.pyplot as pl

    pl.plot(data, label='original data')

    cuda_res = gpu_correlate.dacorrelate(data, mode="same") / data.size
    # print res2

    numpy_res = np.correlate(data, data, mode='same') / data.size
    # print res3

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
    cProfile.run('gpu_correlate.dacorrelate(signal, mode="same")')

    time_n = []
    time_c = []
    time_cs = []

    tlen = []
    for l in range(100, 200000, 50000):
        # signal = np.sin(np.arange(3000000)*0.1) + np.cos(np.arange(3000000)*0.1)*1j
        signal = np.random.rand(l) + np.random.rand(l)*1j


        signal = np.array(signal, dtype='complex64')

        cProfile.run('gpu_correlate.acorrelate(signal, mode="full")', 'restats_cs')
        cProfile.run('np.correlate(signal, signal, mode="full")', 'restats_n')
        cProfile.run('gpu_correlate.dacorrelate(signal, mode="full")', 'restats_c')


        p_n = pstats.Stats('restats_n')
        p_c = pstats.Stats('restats_c')
        p_cs = pstats.Stats('restats_cs')

        print ("{} {} {} {}".format(l,
                         p_n.total_tt/ p_n.total_calls,
                         p_c.total_tt/ p_c.total_calls,
                         p_cs.total_tt/ p_cs.total_calls))

        time_n.append(p_n.total_tt/p_n.total_calls)
        time_c.append(p_c.total_tt/p_c.total_calls)
        time_cs.append(p_cs.total_tt/p_cs.total_calls)

        tlen.append(l)

    pl.plot(tlen, time_n, label='numpy', marker='o')
    pl.plot(tlen, time_c, label='cuda_double', marker='o', color='r')
    pl.plot(tlen, time_cs, label='cuda_single', marker='o', color='g')
    pl.xlabel('number of samples')
    pl.ylabel('time [s]')
    pl.legend()

    pl.show()

