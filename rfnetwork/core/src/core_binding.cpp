
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

#define DATA_NDIM 3

#define COMPLEX_DOUBLE_TYPE 15

void array_data_shape(PyArrayObject* array, int * shape) 
{
    int ndim = PyArray_NDIM(array);           // Number of dimensions
    npy_intp * npy_shape = PyArray_SHAPE(array);    // Pointer to dimensions array

    if (ndim != DATA_NDIM)
    {
        throw std::runtime_error("Invalid data array. Wrong number of dimensions.");
    }

    if (PyArray_TYPE(array) != COMPLEX_DOUBLE_TYPE)
    {
        throw std::runtime_error("Invalid data array. Must be complex double type.");
    }

    if (!(PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS))
    {
        throw std::runtime_error("Invalid data array. Must row ordered (C-style)");
    }

    for (int i = 0; i < DATA_NDIM; ++i) {
        shape[i] = (int) npy_shape[i];
    }

}

static PyObject * connection_matrix_bind(PyObject *self, PyObject *args)
{
    PyObject * s1;
    PyObject * s2;
    PyObject * connections;
    PyObject * probes;
    PyObject * m1;
    PyObject * m2;
    PyObject * row_order;

    if (!PyArg_ParseTuple(args, "OOOOOOO", &s1, &s2, &connections, &probes, &m1, &m2, &row_order))
        return PyLong_FromLong(1);

    int s1_shape[DATA_NDIM];
    int s2_shape[DATA_NDIM];
    int m1_shape[DATA_NDIM];
    int m2_shape[DATA_NDIM];

    int n_connections;

    PyArrayObject* s1_array = (PyArrayObject*) s1;
    array_data_shape(s1_array, s1_shape);

    PyArrayObject* s2_array = (PyArrayObject*) s2;
    array_data_shape(s2_array, s2_shape);

    PyArrayObject* m1_array = (PyArrayObject*) m1;
    array_data_shape(m1_array, m1_shape);

    PyArrayObject* m2_array = (PyArrayObject*) m2;
    array_data_shape(m2_array, m2_shape);

    PyArrayObject* connections_array = (PyArrayObject*) connections;
    n_connections = (int) PyArray_SHAPE(connections_array)[0];

    PyArrayObject* probes_array = (PyArrayObject*) probes;

    PyArrayObject* row_order_array = (PyArrayObject*) row_order;

    int n_row;
    int f_len = s1_shape[0];

    int s1_b = s1_shape[1];
    int s1_a = s1_shape[2];

    int s2_b = s2_shape[1];
    int s2_a = s2_shape[2];

    int b_len = s1_b + s2_b;
    int a_len = s1_a + s2_a;

    // error checking
    // first dimension (frequency) must all be the same size
    if ((s2_shape[0] != f_len) || (m1_shape[0] != f_len) || (m2_shape[0] != f_len))
    {
        throw std::runtime_error("Invalid data array. Unequal sizes in first dimension.");
    }

    // row size of m1 and m2 must equal sum of rows of s1 and s2
    if ((m1_shape[1] != b_len) || (m2_shape[1] != b_len))
    {
        throw std::runtime_error("Invalid row length for m1 and m2.");
    }

    // column size of m1  must equal row size
    if ((m1_shape[2] != b_len))
    {
        throw std::runtime_error("Invalid column length for m1.");
    }

    // column size of m2  must equal row size minus a column for each connection
    if ((m2_shape[2] != (a_len - (2 * n_connections))))
    {
        throw std::runtime_error("Invalid column length for m2.");
    }
    
    // check shape of connections and probes match
    if ((int) PyArray_SHAPE(connections_array)[0] != n_connections)
    {
        throw std::runtime_error("Invalid probe array. Number of connections do not match.");
    }

    connection_matrix(
        (char * ) PyArray_DATA(s1_array),
        (char * ) PyArray_DATA(s2_array),
        (char * ) PyArray_DATA(connections_array),
        (char * ) PyArray_DATA(probes_array),
        (char * ) PyArray_DATA(m1_array),
        (char * ) PyArray_DATA(m2_array),
        (char * ) PyArray_DATA(row_order_array),
        &n_row, f_len, s1_b, s1_a, s2_b, s2_a, n_connections
    );

    return PyLong_FromLong(0);
}

static PyObject * cascade_ndata_bind(PyObject *self, PyObject *args)
{
    PyObject * m1;
    PyObject * m2;
    PyObject * c1;
    PyObject * c2;
    PyObject * out;

    if (!PyArg_ParseTuple(args, "OOOOO", &m1, &m2, &c1, &c2, &out))
        return PyLong_FromLong(1);

    int m1_shape[DATA_NDIM];
    int m2_shape[DATA_NDIM];
    int c1_shape[DATA_NDIM];
    int c2_shape[DATA_NDIM];
    int out_shape[DATA_NDIM];

    PyArrayObject* m1_array = (PyArrayObject*) m1;
    array_data_shape(m1_array, m1_shape);

    PyArrayObject* m2_array = (PyArrayObject*) m2;
    array_data_shape(m2_array, m2_shape);

    PyArrayObject* c1_array = (PyArrayObject*) c1;
    array_data_shape(c1_array, c1_shape);

    PyArrayObject* c2_array = (PyArrayObject*) c2;
    array_data_shape(c2_array, c2_shape);

    PyArrayObject* out_array = (PyArrayObject*) out;
    array_data_shape(out_array, out_shape);

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


    cascade_ndata(
        (char * ) PyArray_DATA(m1_array),
        (char * ) PyArray_DATA(m2_array),
        (char * ) PyArray_DATA(c1_array),
        (char * ) PyArray_DATA(c2_array),
        (char * ) PyArray_DATA(out_array),
        flen, m1_a, m2_a
    );

    return PyLong_FromLong(0);
}

static PyObject * cascade_self_ndata_bind(PyObject *self, PyObject *args)
{
    PyObject * m1;
    PyObject * c1;
    PyObject * out;

    if (!PyArg_ParseTuple(args, "OOO", &m1, &c1, &out))
        return PyLong_FromLong(1);

    int m1_shape[DATA_NDIM];
    int c1_shape[DATA_NDIM];
    int out_shape[DATA_NDIM];

    PyArrayObject* m1_array = (PyArrayObject*) m1;
    array_data_shape(m1_array, m1_shape);

    PyArrayObject* c1_array = (PyArrayObject*) c1;
    array_data_shape(c1_array, c1_shape);

    PyArrayObject* out_array = (PyArrayObject*) out;
    array_data_shape(out_array, out_shape);

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

    cascade_self_ndata(
        (char * ) PyArray_DATA(m1_array),
        (char * ) PyArray_DATA(c1_array),
        (char * ) PyArray_DATA(out_array),
        flen, m1_a
    );

    return PyLong_FromLong(0);
}


static PyMethodDef moduleMethods[] = {
    {"connection_matrix",  connection_matrix_bind, METH_VARARGS, ""},
    {"cascade_ndata",  cascade_ndata_bind, METH_VARARGS, ""},
    {"cascade_self_ndata",  cascade_self_ndata_bind, METH_VARARGS, ""},
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