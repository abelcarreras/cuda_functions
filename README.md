[![PyPI version](https://badge.fury.io/py/cuda_functions.svg)](https://pypi.python.org/pypi/cuda_functions)

CUDA functions
==============
This is a collection of python functions written with CUDA,
using cuFFT and cuBLAS libraries.
The interface with Python is written using the Python C API.

These functions intend to mimic the behavior of numpy functions: fft and correlate
using the power of GPU.


Included functions
---------------------------------------------------------

1. Fast Fourier transform (Complex only)
  - cuda_fft   (single/double precision Fourier transfom)
  - cuda_ifft  (single/double precision inverse Fourier transfom)

2. Autocorrelation functions (Real and complex)
  - cuda_acorrelate (single/double precision autocorrelation function)


Installation
---------------------------------------------------------

1. Run setup.py script to compile only (testing)
   <br>python setup.py build_ext  --inplace

2. To install the module on your system use
   distutils setup.py as usual:
   <br>python setup.py install --user

3. Now module is available on PyPI
   <br>pip install cuda_functions
 
4. Run and check included python scripts as example


Contact info
---------------------------------------------------------

Abel Carreras
<br>abelcarreras83@gmail.com

Department of Materials Science and Engineering
<br>Kyoto University