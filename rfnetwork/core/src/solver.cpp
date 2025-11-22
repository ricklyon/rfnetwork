
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <cstdint>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cstring>
#include <complex>
#include <random>
#include <math.h>
#include <iostream>
#include <thread>

#include "solver.h"

#include "Eigen/Dense"

using Eigen::MatrixXd;

typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixFloatType;

#define DATA_NDIM 3

Field_Ex Ex;
Field_Ey Ey;
Field_Ez Ez;

Field_Hx Hx;
Field_Hy Hy;
Field_Hz Hz;

Coeff_Ex Cx;
Coeff_Ey Cy;
Coeff_Ez Cz;

Coeff_Hx Dx;
Coeff_Hy Dy;
Coeff_Hz Dz;

// get the array of the given name from a python dictionary. Validate the shape
// matches Nx, Ny, Nz
float * get_solver_array(PyObject * dict, const char * name, int Nx, int Ny, int Nz) 
{
    PyObject* py_arr = PyDict_GetItemString(dict, name);
    PyArrayObject* array = (PyArrayObject*) py_arr;
    // get array shape
    npy_intp * npy_shape = PyArray_SHAPE(array);  

    std::ostringstream oss;

    if (PyArray_NDIM(array) != DATA_NDIM)
    {
        throw std::runtime_error("Invalid data array. Wrong number of dimensions.");
    }

    if (PyArray_TYPE(array) != NPY_FLOAT)
    {
        throw std::runtime_error("Invalid data array. Must be float type.");
    }

    if (!(PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS))
    {
        oss << "Invalid data array " << name << ". Must be row ordered (C-style)";
        throw std::runtime_error(oss.str());
    }

    int expected_shape[DATA_NDIM] = {Nx, Ny, Nz};
    // check shape matches expected values
    for (int i = 0; i < DATA_NDIM; i++)
    {
        if (((int) npy_shape[i]) != expected_shape[i])
        {
            oss << "Invalid data array " << name << ". Expected shape on axis " << i << " of " << Nx;
            throw std::runtime_error(oss.str());
        }
    }

    return (float *) PyArray_DATA(array);
}

int solver_init(PyObject * fields, PyObject * coefficients, int Nx, int Ny, int Nz, int Nt)
{
    // initialize ex pointers
    Ex.Nx = Nx;
    Ex.Ny = Ny + 1;
    Ex.Nz = Nz + 1;
    Ex.ex_y = get_solver_array(fields, "ex_y", Ex.Nx, Ex.Ny, Ex.Nz);
    Ex.ex_z = get_solver_array(fields, "ex_z", Ex.Nx, Ex.Ny, Ex.Nz);
    Ex.ex   = get_solver_array(fields, "ex",   Ex.Nx, Ex.Ny, Ex.Nz);
    Ex.NyNz = Ex.Ny * Ex.Nz;

    // y and z endpoints do not get updated and don't have coefficients 
    Cx.Nx = Nx; 
    Cx.Ny = Ny - 1; 
    Cx.Nz = Nz - 1;
    Cx.Cb_ex_y = get_solver_array(coefficients, "Cb_ex_y", Cx.Nx, Cx.Ny, Cx.Nz);
    Cx.Cb_ex_z = get_solver_array(coefficients, "Cb_ex_z", Cx.Nx, Cx.Ny, Cx.Nz);
    Cx.Ca_ex_y = get_solver_array(coefficients, "Ca_ex_y", Cx.Nx, Cx.Ny, Cx.Nz);
    Cx.Ca_ex_z = get_solver_array(coefficients, "Ca_ex_z", Cx.Nx, Cx.Ny, Cx.Nz);
    Cx.NyNz = Cx.Ny * Cx.Nz;

    // initialize ey pointers
    Ey.Nx = Nx + 1;
    Ey.Ny = Ny;
    Ey.Nz = Nz + 1;
    Ey.ey_z = get_solver_array(fields, "ey_z", Ey.Nx, Ey.Ny, Ey.Nz);
    Ey.ey_x = get_solver_array(fields, "ey_x", Ey.Nx, Ey.Ny, Ey.Nz);
    Ey.ey   = get_solver_array(fields, "ey",   Ey.Nx, Ey.Ny, Ey.Nz);
    Ey.NyNz = Ey.Ny * Ey.Nz;

    // x and z endpoints do not get updated and don't have coefficients 
    Cy.Nx = Nx - 1; 
    Cy.Ny = Ny; 
    Cy.Nz = Nz - 1;
    Cy.Cb_ey_z = get_solver_array(coefficients, "Cb_ey_z", Cy.Nx, Cy.Ny, Cy.Nz);
    Cy.Cb_ey_x = get_solver_array(coefficients, "Cb_ey_x", Cy.Nx, Cy.Ny, Cy.Nz);
    Cy.Ca_ey_z = get_solver_array(coefficients, "Ca_ey_z", Cy.Nx, Cy.Ny, Cy.Nz);
    Cy.Ca_ey_x = get_solver_array(coefficients, "Ca_ey_x", Cy.Nx, Cy.Ny, Cy.Nz);
    Cy.NyNz = Cy.Ny * Cy.Nz;

    // initialize ez pointers
    Ez.Nx = Nx + 1;
    Ez.Ny = Ny + 1;
    Ez.Nz = Nz;
    Ez.ez_x = get_solver_array(fields, "ez_x", Ez.Nx, Ez.Ny, Ez.Nz);
    Ez.ez_y = get_solver_array(fields, "ez_y", Ez.Nx, Ez.Ny, Ez.Nz);
    Ez.ez   = get_solver_array(fields, "ez",   Ez.Nx, Ez.Ny, Ez.Nz);
    Ez.NyNz = Ez.Ny * Ez.Nz;

    // x and y endpoints do not get updated and don't have coefficients 
    Cz.Nx = Nx - 1; 
    Cz.Ny = Ny - 1; 
    Cz.Nz = Nz;
    Cz.Cb_ez_x = get_solver_array(coefficients, "Cb_ez_x", Cz.Nx, Cz.Ny, Cz.Nz);
    Cz.Cb_ez_y = get_solver_array(coefficients, "Cb_ez_y", Cz.Nx, Cz.Ny, Cz.Nz);
    Cz.Ca_ez_x = get_solver_array(coefficients, "Ca_ez_x", Cz.Nx, Cz.Ny, Cz.Nz);
    Cz.Ca_ez_y = get_solver_array(coefficients, "Ca_ez_y", Cz.Nx, Cz.Ny, Cz.Nz);
    Cz.NyNz = Cz.Ny * Cz.Nz;

    // initialize hx pointers
    Hx.Nx = Nx + 1;
    Hx.Ny = Ny;
    Hx.Nz = Nz;
    Hx.hx_y = get_solver_array(fields, "hx_y", Hx.Nx, Hx.Ny, Hx.Nz);
    Hx.hx_z = get_solver_array(fields, "hx_z", Hx.Nx, Hx.Ny, Hx.Nz);
    Hx.hx   = get_solver_array(fields, "hx",   Hx.Nx, Hx.Ny, Hx.Nz);
    Hx.NyNz = Ez.Ny * Ez.Nz;

    Dx.Db_hx_y = get_solver_array(coefficients, "Db_hx_y", Hx.Nx, Hx.Ny, Hx.Nz);
    Dx.Db_hx_z = get_solver_array(coefficients, "Db_hx_z", Hx.Nx, Hx.Ny, Hx.Nz);
    Dx.Da_hx_y = get_solver_array(coefficients, "Da_hx_y", Hx.Nx, Hx.Ny, Hx.Nz);
    Dx.Da_hx_z = get_solver_array(coefficients, "Da_hx_z", Hx.Nx, Hx.Ny, Hx.Nz);

    // initialize hy pointers
    Hy.Nx = Nx;
    Hy.Ny = Ny + 1;
    Hy.Nz = Nz;
    Hy.hy_z = get_solver_array(fields, "hy_z", Hy.Nx, Hy.Ny, Hy.Nz);
    Hy.hy_x = get_solver_array(fields, "hy_x", Hy.Nx, Hy.Ny, Hy.Nz);
    Hy.hy   = get_solver_array(fields, "hy",   Hy.Nx, Hy.Ny, Hy.Nz);
    Hy.NyNz = Hy.Ny * Hy.Nz;

    Dy.Db_hy_z = get_solver_array(coefficients, "Db_hy_z", Hy.Nx, Hy.Ny, Hy.Nz);
    Dy.Db_hy_x = get_solver_array(coefficients, "Db_hy_x", Hy.Nx, Hy.Ny, Hy.Nz);
    Dy.Da_hy_z = get_solver_array(coefficients, "Da_hy_z", Hy.Nx, Hy.Ny, Hy.Nz);
    Dy.Da_hy_x = get_solver_array(coefficients, "Da_hy_x", Hy.Nx, Hy.Ny, Hy.Nz);

    // initialize hz pointers
    Hz.Nx = Nx;
    Hz.Ny = Ny;
    Hz.Nz = Nz + 1;
    Hz.hz_x = get_solver_array(fields, "hz_x", Hz.Nx, Hz.Ny, Hz.Nz);
    Hz.hz_y = get_solver_array(fields, "hz_y", Hz.Nx, Hz.Ny, Hz.Nz);
    Hz.hz   = get_solver_array(fields, "hz",   Hz.Nx, Hz.Ny, Hz.Nz);
    Hz.NyNz = Hz.Ny * Hz.Nz;

    Dz.Db_hz_x = get_solver_array(coefficients, "Db_hz_x", Hz.Nx, Hz.Ny, Hz.Nz);
    Dz.Db_hz_y = get_solver_array(coefficients, "Db_hz_y", Hz.Nx, Hz.Ny, Hz.Nz);
    Dz.Da_hz_x = get_solver_array(coefficients, "Da_hz_x", Hz.Nx, Hz.Ny, Hz.Nz);
    Dz.Da_hz_y = get_solver_array(coefficients, "Da_hz_y", Hz.Nx, Hz.Ny, Hz.Nz);

    solver_update_ex(0, Ex.Nx);

    return 0;
}

// update Ex components, starting with the x-axis index x_start, and ending with x_stop (stop index is not inclusive)
int solver_update_ex(int x_start, int x_stop)
{
    // operate on a single slice of the field on the x axis
    int x_offset;
    // number of updated field components
    int Ny = Ex.Ny;
    int Nz = Ex.Nz;

    for (int x = x_start; x < x_stop; x++)
    {   
        x_offset = x * Ex.NyNz;
        MatrixFloatType ex_y (Ex.ex_y + x_offset, Ex.Ny, Ex.Nz);
        MatrixFloatType ex_z (Ex.ex_z + x_offset, Ex.Ny, Ex.Nz);
        MatrixFloatType ex   (Ex.ex   + x_offset, Ex.Ny, Ex.Nz);
        
        x_offset = x * Hz.NyNz;
        MatrixFloatType hz (Hz.hz + x_offset, Hz.Ny, Hz.Nz);
        MatrixFloatType hy (Hy.hy + x_offset, Hy.Ny, Hy.Nz);
        
        // ex coefficients
        x_offset = x * Cx.NyNz;
        MatrixFloatType Cb_ex_y (Cx.Cb_ex_y + x_offset, Ny, Nz);
        MatrixFloatType Cb_ex_z (Cx.Cb_ex_z + x_offset, Ny, Nz);

        MatrixFloatType Ca_ex_y (Cx.Ca_ex_y + x_offset, Ny, Nz);
        MatrixFloatType Ca_ex_z (Cx.Ca_ex_z + x_offset, Ny, Nz);
        
        // update ex_y
        // ex_y[:, 1:-1, 1:-1] = (Ca_ex_y * ex_y[:, 1:-1, 1:-1]) + ex_yd
        ex_y.block(1, 1, Ny, Nz) = (ex_y.block(1, 1, Ny, Nz).array() * Ca_ex_y.array()).matrix();

        // difference terms for hz along y
        // ex_yd = Cb_ex_y * np.diff(hz, axis=1)[:, :, 1:-1]
        ex_y.block(1, 1, Ny, Nz) += (
            Cb_ex_y.array() * (hz.bottomRows(Ny) - hz.topRows(Ny)).middleCols(1, Nz-1).array()
        ).matrix();

        // update ex_z
        // ex_z[:, 1:-1, 1:-1] = (Ca_ex_z * ex_z[:, 1:-1, 1:-1]) + ex_zd
        ex_z.block(1, 1, Ny, Nz) = (ex_z.block(1, 1, Ny -1, Nz -1).array() * Ca_ex_z.array()).matrix();

        // difference terms for hy along z
        // ex_zd = Cb_ex_z * np.diff(hy, axis=2)[:, 1:-1, :]
        ex_z.block(1, 1, Ny, Nz) += (
            Cb_ex_z.array() * (hy.rightCols(Nz) - hy.leftCols(Nz)).middleRows(1, Ny-1).array()
        ).matrix();

        // combine split components
        ex = ex_y + ex_z;
    }

    return 0;
}

// // update Ey components, starting with the x-axis index x_start, and ending with x_stop (stop index is not inclusive)
// int solver_update_ey(SolverConfig * sc, int x_start, int x_stop)
// {
//     int Nx = sc->Nx;
//     int Ny = sc->Ny;
//     int Nz = sc->Nz;

//     float * ey_z_p = sc->field[solver_field_index("ey_z")];
//     float * ey_x_p = sc->field[solver_field_index("ey_x")];
//     float * ey_p = sc->field[solver_field_index("ey")];

//     float * hx_p = sc->field[solver_field_index("hx")];
//     float * hz_p = sc->field[solver_field_index("hz")];
    
//     float * Cb_ey_z_p = sc->field[solver_coeff_index("Cb_ey_z")];
//     float * Cb_ey_x_p = sc->field[solver_coeff_index("Cb_ey_x")];

//     float * Ca_ey_z_p = sc->field[solver_coeff_index("Ca_ey_z")];
//     float * Ca_ey_x_p = sc->field[solver_coeff_index("Ca_ey_x")];

//     // get a single slice of the field on the x axis
//     for (int x = x_start; x < x_stop; x++)
//     {   
//         // endpoints of ey along x do not get updated
//         if ((x < 1) || (x >= Nx))
//         {
//             continue;
//         }

//         MatrixFloatType ey_z (ey_z_p + (x * (Ny) * (Nz + 1)), Ny, Nz + 1);
//         MatrixFloatType ey_x (ey_x_p + (x * (Ny) * (Nz + 1)), Ny, Nz + 1);
//         MatrixFloatType ey   (ey_p   + (x * (Ny) * (Nz + 1)), Ny, Nz + 1);

//         // hx is in the same x plane as ey
//         MatrixFloatType hx (hx_p + (x * (Ny) * (Nz)), Ny, Nz);
        
//         // get hz components on either side of x-slice
//         MatrixFloatType hz_0 (hz_p + ((x - 1) * (Ny) * (Nz + 1)), Ny, Nz + 1);
//         MatrixFloatType hz_1 (hz_p + (x * (Ny) * (Nz + 1)), Ny, Nz + 1);

//         // ey coefficients, endpoints along x and z do not get updated
//         MatrixFloatType Cb_ey_z (Cb_ey_z_p + (x * (Ny) * (Nz - 1)), Ny, Nz - 1);
//         // MatrixFloatType Cb_ex_z (Cb_ex_z_p + (x * (Ny - 1) * (Nz - 1)), Ny - 1, Nz - 1);

//         // MatrixFloatType Ca_ex_y (Ca_ex_y_p + (x * (Ny - 1) * (Nz - 1)), Ny - 1, Nz - 1);
//         // MatrixFloatType Ca_ex_z (Ca_ex_z_p + (x * (Ny - 1) * (Nz - 1)), Ny - 1, Nz - 1);
        
//     }
// }
