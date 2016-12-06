CUDA functions
==============
This is a collection of python functions written with CUDA,
using cuFFT and cuBLAS libraries.
The interface with Python is written using the Python C API.

These functions intend to mimic the behavior of numpy functions: fft and correlate
using the power of GPU.

Included functions
---------------------------------------------------------

1. Fast Fourier transform
  - fft   (single precision Fourier transfom)
  - dfft  (double precision Fourier transfom)
  - ifft  (single precision inverse Fourier transfom)
  - difft (double precision inverse Fourier transfom)

2. Autocorrelation functions
  - acorrelate (single precision autocorrelation)
  - dacorrelate (double precision autocorrelation)


Installation
---------------------------------------------------------

1. Run setup.py script to compile
   python setup.py install build_ext  --inplace

2. Run and check included python scripts as example


Interface
---------------------------------------------------------

- **gpu_fft.dfft**(a)
<br>Parameters    a: array_like (complex64)
<br>Returns       out: array_like (complex64)

- **gpu_fft.dfft**(a)
<br>Parameters    a: array_like (complex128)
<br>Returns       out: array_like (complex128)

- **gpu_fft.ifft**(a)
<br>Parameters    a: array_like (complex64)
<br>Returns:      out: array_like (complex64)

- **gpu_fft.difft**(a)
<br>Parameters    a: array_like (complex128)
<br>Returns       out: array_like (complex128)

- **gpu_correlate.acorrelate**(a, mode='valid')
<br>Parameters    a: array_like (complex64)
              <br>mode: {'valid', 'same', 'full'}, optional
<br>Returns       out: array_like (complex64)

- **gpu_correlate.dacorrelate**(a, mode='valid')
<br>Parameters    a: array_like (complex128)
              <br>mode: {'valid', 'same', 'full'}, optional
<br>Returns       out: array_like (complex128)



Contact info
---------------------------------------------------------
Abel Carreras
<br>abelcarreras83@gmail.com

Department of Materials Science and Engineering
<br>Kyoto University