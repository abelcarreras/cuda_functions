#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cublas_v2.h>


// Define macro functions
#define M_CONC(A, B) M_CONC_(A, B)
#define M_CONC_(A, B) A##B

#define STRINGIFY(x) STRINGIFY_(x)
#define STRINGIFY_(x) #x

// Select precision
#ifdef singleprecision
 #define MODULE_LABEL singleprecision
 typedef float PREC_TYPE;
 #define cublasdotc(...) cublasSdot(__VA_ARGS__);
 #define NPY_CPREC NPY_FLOAT

 float conjugate(float arg)
{
    return arg;
}

#endif

#ifdef doubleprecision
 #define MODULE_LABEL doubleprecision
 typedef double PREC_TYPE;
 #define cublasdotc(...) cublasDdot(__VA_ARGS__);
 #define NPY_CPREC NPY_DOUBLE

 double conjugate(double arg)
{
    return arg;
}

#endif

#ifdef singleprecisioncomplex
 #define MODULE_LABEL singleprecisioncomplex
 typedef float2 PREC_TYPE;
 #define cublasdotc(...) cublasCdotc(__VA_ARGS__);
 #define NPY_CPREC NPY_CFLOAT

float2 conjugate(float2 arg)
{
    return make_float2(arg.x, -arg.y);
}

#endif

#ifdef doubleprecisioncomplex
 #define MODULE_LABEL doubleprecisioncomplex
 typedef double2 PREC_TYPE;
 #define cublasdotc(...) cublasZdotc(__VA_ARGS__);
 #define NPY_CPREC NPY_CDOUBLE

double2 conjugate(double2 arg)
{
    return make_double2(arg.x, -arg.y);
}

#endif

// Init function names
#define MODULE_NAME STRINGIFY(MODULE_LABEL)
#define INIT_FUNCTION M_CONC(init, MODULE_LABEL)
#define INIT_FUNCTION3 M_CONC(PyInit_, MODULE_LABEL)

// complex math functions
static PyObject* autocorrelation(PyObject* self, PyObject *arg, PyObject *keywords);




//  Python Interface
static char function_docstring[] =
    "autocorrelation(signal)\nAutocorrelation function implemented in CUDA";

static PyMethodDef extension_funcs[] = {
    {"acorrelate", (PyCFunction)autocorrelation, METH_VARARGS|METH_KEYWORDS, function_docstring},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_gpu_correlate",
  "This a module that contains a python interface to calculate correlation using cuBLAS",
  -1,
  extension_funcs,
  NULL,
  NULL,
  NULL,
  NULL,
};
#endif


static PyObject *
moduleinit(void)
{
    PyObject *m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3(MODULE_NAME,
        extension_funcs, "acorrelate module");
#endif

  return m;
}

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC
    INIT_FUNCTION(void)
    {
        import_array();
        moduleinit();
    }
#else
    PyMODINIT_FUNC
    INIT_FUNCTION3(void)
    {
        import_array();
        return moduleinit();
    }

#endif


static PyObject* autocorrelation(PyObject* self, PyObject *arg, PyObject *keywords)
{

    const char *mode = "valid";   // Default value of mode (to mimic numpy behavior)

    //  Interface with Python
    PyObject *h_signal_obj;

    static char *kwlist[] = {"input_data", "mode", NULL};
    if (!PyArg_ParseTupleAndKeywords(arg, keywords, "O|s", kwlist, &h_signal_obj, &mode))  return NULL;


    PyObject *h_signal_array = PyArray_FROM_OTF(h_signal_obj, NPY_CPREC, NPY_IN_ARRAY);

    if (h_signal_array == NULL ) {
         Py_XDECREF(h_signal_array);
         return NULL;
    }

    PREC_TYPE *h_signal = (PREC_TYPE *)PyArray_DATA(h_signal_array);
    int     signal_size = (int)PyArray_DIM(h_signal_array, 0);

    PREC_TYPE h_output;

    // Allocate device memory for signal
    PREC_TYPE* d_signal;
    cudaMalloc((void**)&d_signal, sizeof(PREC_TYPE) * signal_size);

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, sizeof(PREC_TYPE) * signal_size, cudaMemcpyHostToDevice);

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Prepare output python object
    PyArrayObject *return_object;

    if  (strcmp(mode, "full") == 0) {

        int dims[1]={signal_size*2-1};
        return_object = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_CPREC);
        PREC_TYPE *return_data  = (PREC_TYPE *)PyArray_DATA(return_object);

        for (int i=0; i< signal_size; i++){
            cublasdotc(handle, signal_size-i,
                         &d_signal[i], 1,
                         d_signal, 1,
                         &h_output);
            return_data[(signal_size*2-1)/2-i] = h_output;
            if (((signal_size*2-1)/2+i) < signal_size*2-1) return_data[(signal_size*2-1)/2+i] = conjugate(h_output);
        }
    }
    else if  (strcmp(mode, "same") == 0) {

        int dims[1]={signal_size};
        return_object = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_CPREC);
        PREC_TYPE *return_data  = (PREC_TYPE *)PyArray_DATA(return_object);

        for (int i=0; i< signal_size/2+1; i++){
            cublasdotc(handle, signal_size-i,
                         &d_signal[i], 1,
                         d_signal, 1,
                         &h_output);
            return_data[signal_size/2-i] = h_output;
            if ((signal_size/2+i) < signal_size) return_data[signal_size/2+i] = conjugate(h_output);
        }
    }
    else if  (strcmp(mode, "valid") == 0) {

        int dims[1]={1};
        return_object = (PyArrayObject *) PyArray_FromDims(1,dims,NPY_CPREC);
        PREC_TYPE *return_data  = (PREC_TYPE *)PyArray_DATA(return_object);

        cublasdotc(handle, signal_size,
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

