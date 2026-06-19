
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include "connect.h"
#include "solver.h"
#include "postprocess.h"

#include <stdio.h>
#include <cstdint>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <complex>
#include <stdexcept>

#define DATA_NDIM 3

static PyObject* solver_run_cu(PyObject* self, PyObject* args) {

    PyObject *coefficients;
    PyObject *probes;
    PyObject *monitors;
    PyObject *mem;
    
    int Nx;
    int Ny;
    int Nz;
    int Nt;
    int n_threads;
    int update_interval;

    // Parse arguments: expecting a single Python object
    if (!PyArg_ParseTuple(
        args, "OOOOIIIIII", &coefficients, &probes, &monitors, &mem, &Nx, &Ny, &Nz, &Nt, &n_threads, &update_interval
    )) {
        return PyLong_FromLong(1);
    }

    if (!PyDict_Check(coefficients)) {
        PyErr_SetString(PyExc_TypeError, "Expected a coefficients dictionary");
        return PyLong_FromLong(1);
    }

    if (!PyList_Check(monitors)) {
        PyErr_SetString(PyExc_TypeError, "Expected a monitors list");
        return PyLong_FromLong(1);
    }

    if (!PyList_Check(probes)) {
        PyErr_SetString(PyExc_TypeError, "Expected a probes list");
        return PyLong_FromLong(1);
    }

    SolverFDTD s;
    s.solver_init_fields(mem, coefficients, Nx, Ny, Nz, 1);
    s.solver_init_monitors(monitors, Nt, 1);
    s.solver_init_probes(probes, Nt);

    #ifdef CUDA_AVAILABLE
        s.solver_run_cu(Nt);
    #else
        throw std::runtime_error("GPU Solver is not available in the current installation.");
    #endif

    return PyLong_FromLong(0);
}

static PyMethodDef moduleMethods[] = {
    {"solver_run_cu",  solver_run_cu, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "cuda_func",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    moduleMethods
};

PyMODINIT_FUNC PyInit_cuda_func(void)
{
    import_array();  
    Py_Initialize();
    return PyModule_Create(&module);
}