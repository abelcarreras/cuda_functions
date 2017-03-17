#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cufft.h>
#include <cublas_v2.h>


// Define macro functions
#define M_CONC(A, B) M_CONC_(A, B)
#define M_CONC_(A, B) A##B

#define STRINGIFY(x) STRINGIFY_(x)
#define STRINGIFY_(x) #x

// Select precision
#ifdef singleprecisioncomplex
 #define MODULE_LABEL singleprecisioncomplex
 typedef float2 PREC_TYPE;
 typedef float SCALE_TYPE;
 typedef cufftComplex PREC_CUTYPE;
 #define cublasscal(...) cublasCsscal(__VA_ARGS__);
 #define cufftExec(...) cufftExecC2C(__VA_ARGS__);
 #define NPY_CPREC NPY_CFLOAT
 #define CUFFT_PLAN CUFFT_C2C
#endif

#ifdef doubleprecisioncomplex
 #define MODULE_LABEL doubleprecisioncomplex
 typedef double2 PREC_TYPE;
 typedef double SCALE_TYPE;
 typedef cufftDoubleComplex PREC_CUTYPE;
 #define cublasscal(...) cublasZdscal(__VA_ARGS__);
 #define cufftExec(...) cufftExecZ2Z(__VA_ARGS__);
 #define NPY_CPREC NPY_CDOUBLE
 #define CUFFT_PLAN CUFFT_Z2Z
#endif

// Init function names
#define MODULE_NAME STRINGIFY(MODULE_LABEL)
#define INIT_FUNCTION M_CONC(init, MODULE_LABEL)


static PyObject* fft(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* ifft(PyObject* self, PyObject *arg, PyObject *keywords);


///////////////////////////////////////////////
//                                           //
//          Fast Fourier Transform           //
//                                           //
///////////////////////////////////////////////

static PyObject* fft(PyObject* self, PyObject *arg, PyObject *keywords)
{

    // Interface with Python
    PyObject *h_signal_obj;

    static char *kwlist[] = {"input_data", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "O", kwlist, &h_signal_obj))  return NULL;

    PyObject *h_signal_array = PyArray_FROM_OTF(h_signal_obj, NPY_CPREC, NPY_IN_ARRAY);

    if (h_signal_array == NULL ) {
         Py_XDECREF(h_signal_array);
         return NULL;
    }

    PREC_TYPE *h_signal = (PREC_TYPE *)PyArray_DATA(h_signal_array);
    int        signal_size = (int)PyArray_DIM(h_signal_array, 0);


    PyArrayObject *return_object;
    int dims[1] = {signal_size};
    return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CPREC);
    PREC_TYPE *return_data  = (PREC_TYPE *)PyArray_DATA(return_object);


    int mem_size = sizeof(PREC_TYPE) * signal_size;

    // Allocate device memory for signal in the device
    PREC_TYPE* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, mem_size,
               cudaMemcpyHostToDevice);

    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, signal_size, CUFFT_PLAN, 1);

    // Fourier transform using CUFFT_FORWARD
    cufftExec(plan, (PREC_CUTYPE *)d_signal, (PREC_CUTYPE *)d_signal, CUFFT_FORWARD);


    // Copy device memory to host
    cudaMemcpy(return_data, d_signal, mem_size,
               cudaMemcpyDeviceToHost);

    // cleanup memory
    cufftDestroy(plan);
    cudaFree(d_signal);
    Py_DECREF(h_signal_array);

    return(PyArray_Return(return_object));
}


////////////////////////////////////////////////
//                                            //
//      Inverse Fast Fourier Transform        //
//                                            //
////////////////////////////////////////////////

static PyObject* ifft(PyObject* self, PyObject *arg, PyObject *keywords)
{
    //  Interface with Python
    PyObject *h_signal_obj;

    static char *kwlist[] = {"input_data", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "O", kwlist, &h_signal_obj))  return NULL;

    PyObject *h_signal_array = PyArray_FROM_OTF(h_signal_obj, NPY_CPREC, NPY_IN_ARRAY);

    if (h_signal_array == NULL ) {
         Py_XDECREF(h_signal_array);
         return NULL;
    }

    PREC_TYPE *h_signal = (PREC_TYPE *)PyArray_DATA(h_signal_array);
    int     signal_size = (int)PyArray_DIM(h_signal_array, 0);


    //Create new numpy array for storing result
    PyArrayObject *return_object;
    int dims[1]={signal_size};
    return_object = (PyArrayObject *) PyArray_FromDims(1,dims ,NPY_CPREC);
    PREC_TYPE *return_data  = (PREC_TYPE *)PyArray_DATA(return_object);

    int mem_size = sizeof(PREC_TYPE) * signal_size;

    // Allocate device memory for signal in the device
    PREC_TYPE* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, mem_size,
               cudaMemcpyHostToDevice);

    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, signal_size, CUFFT_PLAN, 1);

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Inverse Fourier transform using CUFFT_INVERSE
    cufftExec(plan, (PREC_CUTYPE *)d_signal, (PREC_CUTYPE *)d_signal, CUFFT_INVERSE);

    SCALE_TYPE alpha = 1.0 / signal_size;
    cublasscal(handle, signal_size,
               &alpha,
               d_signal, 1);

    // Copy device memory to host
    cudaMemcpy(return_data, d_signal, mem_size,
               cudaMemcpyDeviceToHost);

    // cleanup memory
    cublasDestroy(handle);
    cufftDestroy(plan);
    cudaFree(d_signal);
    Py_DECREF(h_signal_array);

    return(PyArray_Return(return_object));
}


static char extension_docs_fft[] =
    "fft(signal)\nFast Fourier Transform implemented in CUDA\n using cuFFT\n  ";


static char extension_docs_ifft[] =
    "ifft(signal)\nInverse Fast Fourier Transform implemented in CUDA\n using cuFFT\n  ";


static PyMethodDef extension_funcs[] =
{
    {"fft", (PyCFunction) fft, METH_VARARGS|METH_KEYWORDS, extension_docs_fft},
    {"ifft", (PyCFunction) ifft, METH_VARARGS|METH_KEYWORDS, extension_docs_ifft},
    {NULL}
};


PyMODINIT_FUNC INIT_FUNCTION(void)
{
//  Importing numpy array types
    import_array();
    Py_InitModule3(MODULE_NAME, extension_funcs,
                   "Fast Fourier Tranform functions (CUDA)");
};