
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include "connect.h"

#include <stdio.h>
#include <cstdint>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <complex>
#include <stdexcept>

#define NDATA_NDIM 3

#define COMPLEX_DOUBLE_TYPE 15

void array_ndata_shape(PyArrayObject* array, int * shape) 
{
    int ndim = PyArray_NDIM(array);           // Number of dimensions
    npy_intp * npy_shape = PyArray_SHAPE(array);    // Pointer to dimensions array

    if (ndim != NDATA_NDIM)
    {
        throw std::runtime_error("Invalid noise data array. Wrong number of dimensions.");
    }

    if (PyArray_TYPE(array) != COMPLEX_DOUBLE_TYPE)
    {
        throw std::runtime_error("Invalid noise data array. Must be complex double type.");
    }

    if (!(PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS))
    {
        throw std::runtime_error("Invalid noise data array. Must row ordered (C-style)");
    }

    for (int i = 0; i < NDATA_NDIM; ++i) {
        shape[i] = (int) npy_shape[i];
    }

}

static PyObject * cascade_noise_data(PyObject *self, PyObject *args)
{
    PyObject * m1;
    PyObject * m2;
    PyObject * c1;
    PyObject * c2;
    PyObject * out;

    if (!PyArg_ParseTuple(args, "OOOOO", &m1, &m2, &c1, &c2, &out))
        return PyLong_FromLong(1);

    int m1_shape[NDATA_NDIM];
    int m2_shape[NDATA_NDIM];
    int c1_shape[NDATA_NDIM];
    int c2_shape[NDATA_NDIM];
    int out_shape[NDATA_NDIM];

    PyArrayObject* m1_array = (PyArrayObject*) m1;
    array_ndata_shape(m1_array, m1_shape);

    PyArrayObject* m2_array = (PyArrayObject*) m2;
    array_ndata_shape(m2_array, m2_shape);

    PyArrayObject* c1_array = (PyArrayObject*) c1;
    array_ndata_shape(c1_array, c1_shape);

    PyArrayObject* c2_array = (PyArrayObject*) c2;
    array_ndata_shape(c2_array, c2_shape);

    PyArrayObject* out_array = (PyArrayObject*) out;
    array_ndata_shape(out_array, out_shape);

    // error checking
    // first dimension (frequency) must all be the same size
    int flen = m1_shape[0];
    if ((m2_shape[0] != flen) || (c1_shape[0] != flen) || (c2_shape[0] != flen) || (out_shape[0] != flen))
    {
        throw std::runtime_error("Invalid noise data array. Unequal sizes in first dimension.");
    }

    // column size of m1 must equal row and column size of c1
    int m1_a = m1_shape[2];
    if ((c1_shape[1] != m1_a) || (c1_shape[2] != m1_a))
    {
        throw std::runtime_error("Invalid noise data array for C1.");
    }

    // column size of m1 must equal columns of c1
    int m2_a = m2_shape[2];
    if ((c2_shape[1] != m2_a) || (c2_shape[2] != m2_a))
    {
        throw std::runtime_error("Invalid noise data array for C2.");
    }

    // row size of M arrays must match
    int m_b = m1_a + m2_a;
    if ((m1_shape[1] != m_b) || (m2_shape[1] != m_b) || (out_shape[1] != m_b) || (out_shape[2] != m_b))
    {
        throw std::runtime_error("Invalid noise data array. Unequal sizes in rows of M1/M2.");
    }


    cascade_noise_data(
        (char * ) PyArray_DATA(m1_array),
        (char * ) PyArray_DATA(m2_array),
        (char * ) PyArray_DATA(c1_array),
        (char * ) PyArray_DATA(c2_array),
        (char * ) PyArray_DATA(out_array),
        flen, m1_a, m2_a
    );

    return PyLong_FromLong(0);
}

static PyObject * cascade_self_noise_data(PyObject *self, PyObject *args)
{
    PyObject * m1;
    PyObject * c1;
    PyObject * out;

    if (!PyArg_ParseTuple(args, "OOO", &m1, &c1, &out))
        return PyLong_FromLong(1);

    int m1_shape[NDATA_NDIM];
    int c1_shape[NDATA_NDIM];
    int out_shape[NDATA_NDIM];

    PyArrayObject* m1_array = (PyArrayObject*) m1;
    array_ndata_shape(m1_array, m1_shape);

    PyArrayObject* c1_array = (PyArrayObject*) c1;
    array_ndata_shape(c1_array, c1_shape);

    PyArrayObject* out_array = (PyArrayObject*) out;
    array_ndata_shape(out_array, out_shape);

    // error checking
    // first dimension (frequency) must all be the same size
    int flen = m1_shape[0];
    if ((c1_shape[0] != flen) || (out_shape[0] != flen))
    {
        throw std::runtime_error("Invalid noise data array. Unequal sizes in first dimension.");
    }

    // column size of m1 must equal row and column size of c1
    int m1_a = m1_shape[2];
    if ((c1_shape[1] != m1_a) || (c1_shape[2] != m1_a))
    {
        throw std::runtime_error("Invalid noise data array for C1.");
    }

    cascade_self_noise_data(
        (char * ) PyArray_DATA(m1_array),
        (char * ) PyArray_DATA(c1_array),
        (char * ) PyArray_DATA(out_array),
        flen, m1_a
    );

    return PyLong_FromLong(0);
}


static PyMethodDef moduleMethods[] = {
    {"cascade_noise_data",  cascade_noise_data, METH_VARARGS, ""},
    {"cascade_self_noise_data",  cascade_self_noise_data, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "core_func",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    moduleMethods
};

PyMODINIT_FUNC PyInit_core_func(void)
{
    Py_Initialize();
    return PyModule_Create(&module);
}