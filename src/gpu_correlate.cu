#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cublas_v2.h>

// Complex data type
typedef float2 Complex;
typedef double2 DoubleComplex;


static PyObject* autocorrelation_sp(PyObject* self, PyObject *arg, PyObject *keywords);
static PyObject* autocorrelation_dp(PyObject* self, PyObject *arg, PyObject *keywords);


///////////////////////////////////////////////////////////
//                                                       //
// Single precision autocorrelation (Complex to complex) //
//                                                       //
///////////////////////////////////////////////////////////

static PyObject* autocorrelation_sp(PyObject* self, PyObject *arg, PyObject *keywords)
{
    const char *mode = "valid";   // Default value of mode (to mimic numpy behavior)

    //  Interface with Python
    PyObject *h_signal_obj;

    static char *kwlist[] = {"input_data", "mode", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "O|s", kwlist, &h_signal_obj, &mode))  return NULL;

    PyObject *h_signal_array = PyArray_FROM_OTF(h_signal_obj, NPY_CFLOAT, NPY_IN_ARRAY);

    if (h_signal_array == NULL ) {
         Py_XDECREF(h_signal_array);
         return NULL;
    }

    Complex *h_signal = (Complex *)PyArray_DATA(h_signal_array);
    int     signal_size = (int)PyArray_DIM(h_signal_array, 0);

    Complex h_output;

    // Allocate device memory for signal
    Complex* d_signal;
    cudaMalloc((void**)&d_signal, sizeof(Complex) * signal_size);

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, sizeof(Complex) * signal_size, cudaMemcpyHostToDevice);

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Prepare output python object
    PyArrayObject *return_object;

    if  (strcmp(mode, "full") == 0) {

        int dims[1]={signal_size*2-1};
        return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CFLOAT);
        Complex *return_data  = (Complex *)PyArray_DATA(return_object);

        for (int i=0; i< signal_size; i++){
            cublasCdotc(handle, signal_size-i,
                         &d_signal[i], 1,
                         d_signal, 1,
                         &h_output);
            return_data[(signal_size*2-1)/2-i] = h_output;
            if (((signal_size*2-1)/2+i) < signal_size*2-1) return_data[(signal_size*2-1)/2+i] = h_output;
        }
    }
    else if  (strcmp(mode, "same") == 0) {

        int dims[1]={signal_size};
        return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CFLOAT);
        Complex *return_data  = (Complex *)PyArray_DATA(return_object);

        for (int i=0; i< signal_size/2+1; i++){
            cublasCdotc(handle, signal_size-i,
                         &d_signal[i], 1,
                         d_signal, 1,
                         &h_output);
            return_data[signal_size/2-i] = h_output;
            if ((signal_size/2+i) < signal_size) return_data[signal_size/2+i] = h_output;
        }
    }
    else if  (strcmp(mode, "valid") == 0) {

        int dims[1]={1};
        return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CFLOAT);
        Complex *return_data  = (Complex *)PyArray_DATA(return_object);

        cublasCdotc(handle, signal_size,
                     d_signal, 1,
                     d_signal, 1,
                     &h_output);
        return_data[0] = h_output;
    }
   else {
        PyErr_SetString(PyExc_TypeError, "this mode do not exist");
        PyErr_Print();
    }

    // cleanup memory
    cudaFree(d_signal);
    cublasDestroy(handle);
    Py_DECREF(h_signal_array);

    return(PyArray_Return(return_object));
}


///////////////////////////////////////////////////////////
//                                                       //
// Double precision autocorrelation (Complex to complex) //
//                                                       //
///////////////////////////////////////////////////////////

static PyObject* autocorrelation_dp(PyObject* self, PyObject *arg, PyObject *keywords)
{
    const char    *mode = "valid";   // Default value of mode (to mimic numpy behavior)

    //  Interface with Python
    PyObject *h_signal_obj;

    static char *kwlist[] = {"input_data", "mode", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "O|s", kwlist, &h_signal_obj, &mode))  return NULL;

    PyObject *h_signal_array = PyArray_FROM_OTF(h_signal_obj, NPY_CDOUBLE, NPY_IN_ARRAY);

    if (h_signal_array == NULL ) {
         Py_XDECREF(h_signal_array);
         return NULL;
    }

    DoubleComplex *h_signal = (DoubleComplex *)PyArray_DATA(h_signal_array);
    int     signal_size = (int)PyArray_DIM(h_signal_array, 0);

    DoubleComplex h_output;

    // Allocate device memory for signal
    DoubleComplex* d_signal;
    cudaMalloc((void**)&d_signal, sizeof(DoubleComplex) * signal_size);

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, sizeof(DoubleComplex) * signal_size, cudaMemcpyHostToDevice);

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Prepare output python object
    PyArrayObject *return_object;

    if  (strcmp(mode, "full") == 0) {

        int dims[1]={signal_size*2-1};
        return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CDOUBLE);
        DoubleComplex *return_data  = (DoubleComplex *)PyArray_DATA(return_object);

        for (int i=0; i< signal_size; i++){
            cublasZdotc(handle, signal_size-i,
                         &d_signal[i], 1,
                         d_signal, 1,
                         &h_output);
            if (((signal_size*2-1)/2+i) < signal_size*2-1) return_data[(signal_size*2-1)/2+i] = h_output;
            return_data[(signal_size*2-1)/2-i] = h_output;
        }
    }
    else if  (strcmp(mode, "same") == 0) {

        int dims[1]={signal_size};
        return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CDOUBLE);
        DoubleComplex *return_data  = (DoubleComplex *)PyArray_DATA(return_object);

        for (int i=0; i< signal_size/2+1; i++){
            cublasZdotc(handle, signal_size-i,
                         &d_signal[i], 1,
                         d_signal, 1,
                         &h_output);

            if ((signal_size/2+i) < signal_size) return_data[signal_size/2+i] = h_output;
            return_data[signal_size/2-i] = h_output;
        }

    }
    else if  (strcmp(mode, "valid") == 0) {

        int dims[1]={1};
        return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CDOUBLE);
        DoubleComplex *return_data  = (DoubleComplex *)PyArray_DATA(return_object);

        cublasZdotc(handle, signal_size,
                     d_signal, 1,
                     d_signal, 1,
                     &h_output);
        return_data[0] = h_output;
    }
   else {
        PyErr_SetString(PyExc_TypeError, "this mode do not exist");
        PyErr_Print();
   }

    // Finish cublas
    cublasDestroy(handle);
    cudaFree(d_signal);
    Py_DECREF(h_signal_array);

    return(PyArray_Return(return_object));
}



static char extension_docs_sp[] =
    "autocorrelation(signal)\nAutocorrelation implemented in CUDA\n(Single precision)\n  ";

static char extension_docs_dp[] =
    "autocorrelation(signal)\nAutocorrelation implemented in CUDA\n(Double precision)\n  ";


static PyMethodDef extension_funcs[] =
{
    {"acorrelate", (PyCFunction) autocorrelation_sp, METH_VARARGS|METH_KEYWORDS, extension_docs_sp},
    {"dacorrelate", (PyCFunction) autocorrelation_dp, METH_VARARGS|METH_KEYWORDS, extension_docs_dp},
    {NULL}
};


PyMODINIT_FUNC initgpu_correlate(void)
{
//  Importing numpy array types
    import_array();
    Py_InitModule3("gpu_correlate", extension_funcs,
                   "Autocorrelation functions (CUDA)");
};