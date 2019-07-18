# Adapted from: https://github.com/rmcgibbo/npcuda-example
# Copyright (c) 2014, Robert T. McGibbon and the Authors
# All rights reserved.

try:
    from setuptools import setup, Extension
    use_setuptools = True
    print("setuptools is used.")
except ImportError:
    from distutils.core import setup, Extension
    use_setuptools = False
    print("distutils is used.")


import os
from os.path import join as pjoin
from distutils.command.build_ext import build_ext
import numpy

def get_version_number():
    for l in open('cuda_functions/__init__.py', 'r').readlines():
        if not(l.find('__version__')):
            exec(l, globals())
            return __version__

def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

CUDA = locate_cuda()

include_dirs_numpy = numpy.get_include()

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
    #    cc_args.remove('-fno-strict-aliasing')
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            print (CUDA['nvcc'])
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


# FFT/iFFT functions
fft_module_test = []
for prec, file in [
                   ('doubleprecisioncomplex', 'gpu_fft_dpc'),
                   ('singleprecisioncomplex', 'gpu_fft_spc')
                  ]:
    fft_module_test.append(Extension('cuda_functions.bin.' + file,
                           sources=['src/cuFFT.cu'],
                           include_dirs=[include_dirs_numpy, CUDA['include']],
                           library_dirs=[CUDA['lib64']],
                           runtime_library_dirs=[CUDA['lib64']],
                           libraries=['cudart', 'cufft', 'cublas'],
                           extra_compile_args={'gcc': [],
                                              'nvcc': [
                                                       # '-arch=sm_60',
                                                       '-D'+prec+'='+file,
                                                       '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
                           ))

# Autocorrelation functions
acorr_module_test = []
for prec, file in [
                   ('doubleprecision',        'gpu_correlate_dp'),
                   ('doubleprecisioncomplex', 'gpu_correlate_dpc'),
                   ('singleprecision',        'gpu_correlate_sp'),
                   ('singleprecisioncomplex', 'gpu_correlate_spc')
                  ]:
    acorr_module_test.append(Extension('cuda_functions.bin.' + file,
                             sources=['src/autocorrelation.cu'],
                             include_dirs=[include_dirs_numpy, CUDA['include']],
                             library_dirs=[CUDA['lib64']],
                             runtime_library_dirs=[CUDA['lib64']],
                             libraries=['cudart', 'cublas'],
                             extra_compile_args={'gcc': [],
                                                 'nvcc': [
                                                           # '-arch=sm_60',
                                                           '-D'+prec+'='+file,
                                                           '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
                             ))


setup(name='cuda_functions',
      author='Abel Carreras',
      description='cuda_functions module',
      url='https://github.com/abelcarreras/cuda_functions',
      author_email='abelcarreras83@gmail.com',
      version=get_version_number(),
      ext_modules=fft_module_test + acorr_module_test,
      packages=['cuda_functions',
                'cuda_functions.bin'],
      license='MIT License',
      requires=['numpy'],
      # inject our custom trigger
      cmdclass={'build_ext': custom_build_ext}
      )
