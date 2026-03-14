
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

#include "postprocess.h"

#include "Eigen/Dense"

using Eigen::MatrixXd;

typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixFloatType;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> MatrixFloatStride;
typedef Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixComplexType;

#define FFDATA_NDIM 4

// placement of far-field dims in data array
#define FF_POLARIZATION 0
#define FF_FREQUENCY 1
#define FF_THETA 2
#define FF_PHI 3

std::complex<double> * get_complex_array(PyObject* py_obj, int * shape, int ndim) 
{   
    PyArrayObject* array = (PyArrayObject*) py_obj;
    // get array shape
    npy_intp * npy_shape = PyArray_SHAPE(array);  

    std::ostringstream oss;

    if (ndim != PyArray_NDIM(array))
    {
        throw std::runtime_error("Invalid data array. Wrong number of dimensions.");
    }

    if (PyArray_TYPE(array) != NPY_CDOUBLE)
    {
        throw std::runtime_error("Invalid data array. Must be complex double type.");
    }

    if (!(PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS))
    {
        throw std::runtime_error("Invalid data array. Must row ordered (C-style)");
    }

    for (int i = 0; i < ndim; ++i) {
        if (shape[i] != (int) npy_shape[i])
        {
            throw std::runtime_error("Invalid data source array shape");
        }
    }

    return (std::complex<double> *) PyArray_DATA(array);
}

float * get_float_array(PyObject * py_obj, int * shape, int ndim) 
{   
    PyArrayObject* array = (PyArrayObject*) py_obj;
    // get array shape
    npy_intp * npy_shape = PyArray_SHAPE(array);  

    std::ostringstream oss;

    if (ndim != PyArray_NDIM(array))
    {
        throw std::runtime_error("Invalid data array. Wrong number of dimensions.");
    }

    if (PyArray_TYPE(array) != NPY_FLOAT)
    {
        throw std::runtime_error("Invalid data array. Must be float type.");
    }

    if (!(PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS))
    {
        throw std::runtime_error("Invalid data array. Must row ordered (C-style)");
    }

    for (int i = 0; i < ndim; ++i) {
        if (shape[i] != (int) npy_shape[i])
        {
            throw std::runtime_error("Invalid data source array shape");
        }
    }

    return (float *) PyArray_DATA(array);
}

int get_pointer_index_4(int * shape, const std::array<int, 4>& index)
{
    int idx = 0;
    int n_dim = 4;
    
    for (int i = 0; i < n_dim; i++)
    {
        // compute size of all lower dimensions
        int block_size = 1; 
        for (int j = i + 1; j < n_dim; j++)
        {
            block_size *= shape[j];
        }

        // increment by the number of blocks given by this index, at the last index block_size is 1.
        idx += (index[i] * block_size);
    }

    return idx;
}


int postprocess_nf2ff(
    PyObject * J_xyz_py, 
    PyObject * M_xyz_py, 
    PyObject * r_grid_py, 
    PyObject * w_grid_py, 
    PyObject * surf_pos_py, 
    PyObject * ff_data_py
)
{
    npy_intp * npy_shape;

    PyObject* py_data = PyDict_GetItemString(ff_data_py, "data");
    PyArrayObject* py_data_arr = (PyArrayObject*) py_data;

    if (PyArray_NDIM(py_data_arr) != FFDATA_NDIM)
    {
        throw std::runtime_error("Invalid far-field data array. Expected four dimensions.");
    }

    if (PyList_Size(r_grid_py) != 3)
    {
        throw std::runtime_error("Invalid r_grid data array. Expected list of length 3.");
    }

    // dimensions are polarization, frequency, theta, phi
    int data_shape[FFDATA_NDIM];
    npy_shape = PyArray_SHAPE(py_data_arr); 

    for (int i = 0; i < FFDATA_NDIM; ++i) { 
        data_shape[i] = npy_shape[i];
    }

    // result data array
    std::complex<double> * data_array = get_complex_array(py_data, data_shape, FFDATA_NDIM);

    // beta (frequency) array
    int beta_shape[1] = {data_shape[FF_FREQUENCY]};
    float * beta_arr = get_float_array(
        PyDict_GetItemString(ff_data_py, "beta"), beta_shape, 1
    );

    // theta array, in radius
    int theta_shape[1] = {data_shape[FF_THETA]};
    float * theta_arr = get_float_array(
        PyDict_GetItemString(ff_data_py, "theta"), theta_shape, 1
    );

    // phi array, in radius
    int phi_shape[1] = {data_shape[FF_PHI]};
    float * phi_arr = get_float_array(
        PyDict_GetItemString(ff_data_py, "phi"), phi_shape, 1
    );

    // shape of the grid on each axis
    int grid_shape[3][2];
    // shape of J and M current grids on each axis, currents have dimensions (xyz, frequency, grid_shape)
    int JM_shape[3][4];
    
    // array pointers for J currents, two per axis on each face
    std::complex<double> * J_xyz_p[3][2];
    // array pointers for M currents, two per axis on each face
    std::complex<double> * M_xyz_p[3][2];
    // array pointers for cell positions on each axis
    float * r_grid_p[3][2];
    // array pointers for cell widths on each axis
    float * w_grid_p[3][2];
    // surface positions, 2 values per axis
    float surf_pos[3][2];

    // get temporary working complex array
    PyObject* working_grid_py = PyDict_GetItemString(ff_data_py, "working_grid_cmplx");
    PyArrayObject* working_grid_array = (PyArrayObject*) working_grid_py;
    std::complex<double> * working_grid_cmplx_p = (std::complex<double> *) PyArray_DATA(working_grid_array);

    if (PyArray_TYPE(working_grid_array) != NPY_CDOUBLE)
    {
        throw std::runtime_error("Invalid data array. Must be complex double type.");
    }

    // get temporary working float array
    working_grid_py = PyDict_GetItemString(ff_data_py, "working_grid_float");
    working_grid_array = (PyArrayObject*) working_grid_py;
    float * working_grid_float_p = (float *) PyArray_DATA(working_grid_array);

    if (PyArray_TYPE(working_grid_array) != NPY_FLOAT)
    {
        throw std::runtime_error("Invalid data array. Must be float type.");
    }

    // TODO: check shape of working grid array

    // get array pointers for each current source, cell positions, and widths.
    for (int axis = 0; axis < 3; axis++)
    {
        // get two arrays per axis for x/y positions
        PyObject* r_grid_axis = PyList_GetItem(r_grid_py, axis);
        PyObject* r_grid_axis_1 = PyList_GetItem(r_grid_axis, 0);
        PyObject* r_grid_axis_2 = PyList_GetItem(r_grid_axis, 1);

        // get two arrays per axis for x/y cell widths
        PyObject* w_grid_axis = PyList_GetItem(w_grid_py, axis);
        PyObject* w_grid_axis_1 = PyList_GetItem(w_grid_axis, 0);
        PyObject* w_grid_axis_2 = PyList_GetItem(w_grid_axis, 1);

        // get shape of the grid
        PyArrayObject* w_grid_axis_arr_1 = (PyArrayObject*) w_grid_axis_1;
        npy_shape = PyArray_SHAPE(w_grid_axis_arr_1);
        grid_shape[axis][0] = (int) npy_shape[0];
        grid_shape[axis][1] = (int) npy_shape[1];
        // shape of current arrays
        JM_shape[axis][0] = 3;
        JM_shape[axis][1] = data_shape[FF_FREQUENCY];
        JM_shape[axis][2] = grid_shape[axis][0];
        JM_shape[axis][3] = grid_shape[axis][1];

        // get arrays for the grid positions, values are a meshgrid
        r_grid_p[axis][0] = get_float_array(r_grid_axis_1, grid_shape[axis], 2);
        r_grid_p[axis][1] = get_float_array(r_grid_axis_2, grid_shape[axis], 2);

        // get arrays for the grid cell sizes, values are a meshgrid
        w_grid_p[axis][0] = get_float_array(w_grid_axis_1, grid_shape[axis], 2);
        w_grid_p[axis][1] = get_float_array(w_grid_axis_2, grid_shape[axis], 2);

        // array pointers for the J current
        PyObject * J_xyz_axis = PyList_GetItem(J_xyz_py, axis);
        J_xyz_p[axis][0] = get_complex_array(
            PyList_GetItem(J_xyz_axis, 0), JM_shape[axis], 4
        );
        J_xyz_p[axis][1] = get_complex_array(
            PyList_GetItem(J_xyz_axis, 1), JM_shape[axis], 4
        );

        // array pointers for the M current
        PyObject * M_xyz_axis = PyList_GetItem(M_xyz_py, axis);
        M_xyz_p[axis][0] = get_complex_array(
            PyList_GetItem(M_xyz_axis, 0), JM_shape[axis], 4
        );
        M_xyz_p[axis][1] = get_complex_array(
            PyList_GetItem(M_xyz_axis, 1), JM_shape[axis], 4
        );

        // surface position for each face on axis
        PyObject * surf_pos_axis = PyList_GetItem(surf_pos_py, axis);
        surf_pos[axis][0] = (float) PyFloat_AsDouble(PyList_GetItem(surf_pos_axis, 0));
        surf_pos[axis][1] = (float) PyFloat_AsDouble(PyList_GetItem(surf_pos_axis, 1));

    }
    
    std::complex<double> N_theta, N_phi, L_theta, L_phi;
    std::complex<double> * thetapol_p;
    std::complex<double> * phipol_p;


    float beta, theta, phi;
    double cos_th, cos_ph, sin_th, sin_ph;
    // cos/sin terms cast as complex values
    std::complex<double> cos_th_c, cos_ph_c, sin_th_c, sin_ph_c;

    double u, v, w;
    int grid_size;

    // constant -1j
    std::complex<double> n1J = std::complex<double>(0, -1);
    std::complex<double> zero_c = std::complex<double>(0, 0);

    // loop over theta
    for (int th = 0; th < data_shape[FF_THETA]; th++)
    {
        theta = theta_arr[th];
        // loop over phi
        for (int ph = 0; ph < data_shape[FF_PHI]; ph++)
        {
            phi = phi_arr[ph];
            
            cos_ph = std::cos((double) phi);
            cos_th = std::cos((double) theta);
            sin_ph = std::sin((double) phi);
            sin_th = std::sin((double) theta);

            u = sin_th * cos_ph;
            v = sin_th * sin_ph;
            w = cos_th;

            cos_ph_c = (std::complex<double>) cos_ph;
            cos_th_c = (std::complex<double>) cos_th;
            sin_ph_c = (std::complex<double>) sin_ph;
            sin_th_c = (std::complex<double>) sin_th;

            // loop over frequency
            for (int f = 0; f < data_shape[FF_FREQUENCY]; f++)
            {   
                // reset N and L at each frequency, values from all faces are summed into a single value
                N_theta = zero_c;
                N_phi = zero_c;
                L_theta = zero_c;
                L_phi = zero_c;

                // loop over x, y, z axis
                for (int axis = 0; axis < 3; axis++)
                {
                    // cell positions on the surface
                    MatrixFloatType r_pos_0 (r_grid_p[axis][0], grid_shape[axis][0], grid_shape[axis][1]);
                    MatrixFloatType r_pos_1 (r_grid_p[axis][1], grid_shape[axis][0], grid_shape[axis][1]);
                    
                    // cell widths
                    MatrixFloatType w_0 (w_grid_p[axis][0], grid_shape[axis][0], grid_shape[axis][1]);
                    MatrixFloatType w_1 (w_grid_p[axis][1], grid_shape[axis][0], grid_shape[axis][1]);

                    grid_size = grid_shape[axis][0] * grid_shape[axis][1];

                    // each of the two faces on axis
                    for (int s = 0; s < 2; s++)
                    {
                        // phase term for integrand
                        MatrixFloatType r_dot (working_grid_float_p, grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // (r_pos[0] * u + r_pos[1] * v + r_pos[2] * w))
                        if (axis == 0)
                        {
                            r_dot = ((surf_pos[axis][s] * u) + ((r_pos_0 * v).array() + (r_pos_1 * w).array()));
                        }
                        else if (axis == 1)
                        {
                            r_dot = ((surf_pos[axis][s] * v) + ((r_pos_0 * u).array() + (r_pos_1 * w).array()));
                        }
                        else
                        {
                            r_dot = ((surf_pos[axis][s] * w) + ((r_pos_0 * u).array() + (r_pos_1 * v).array()));
                        }


                        MatrixComplexType intg (working_grid_cmplx_p, grid_shape[axis][0], grid_shape[axis][1]);
                        beta = beta_arr[f];

                        // phs_term = exp(1j * beta.item() * r_dot)
                        intg = ((n1J * ((std::complex<double>) beta)) * r_dot.array().cast<std::complex<double>>()).array().exp();
                        // dS = phs_term * d1 * d2
                        intg = intg.cwiseProduct(w_0.cast<std::complex<double>>());
                        intg = intg.cwiseProduct(w_1.cast<std::complex<double>>());

                        // Jx current grid
                        std::complex<double> * Jx_p = J_xyz_p[axis][s] + get_pointer_index_4(JM_shape[axis], {0, f, 0, 0});
                        MatrixComplexType Jx (Jx_p, grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // Jy current grid
                        std::complex<double> * Jy_p = (J_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {1, f, 0, 0});
                        MatrixComplexType Jy (Jy_p, grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // Jz current grid
                        std::complex<double> * Jz_p = (J_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {2, f, 0, 0});
                        MatrixComplexType Jz (Jz_p, grid_shape[axis][0], grid_shape[axis][1]);

                        // Mx current grid
                        std::complex<double> * Mx_p = (M_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {0, f, 0, 0});
                        MatrixComplexType Mx (Mx_p, grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // My current grid
                        std::complex<double> * My_p = (M_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {1, f, 0, 0});
                        MatrixComplexType My (My_p, grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // Mz current grid
                        std::complex<double> * Mz_p = (M_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {2, f, 0, 0});
                        MatrixComplexType Mz (Mz_p, grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // N_theta
                        // N_theta_intg = (Jx * cos_th * cos_ph) + (Jy * cos_th * sin_ph) - (Jz * sin_th)
                        // N_theta += np.sum(N_theta_intg * dS)
                        N_theta += (
                            (
                                (Jx.array() * (cos_th_c * cos_ph_c)) + 
                                (Jy.array() * (cos_th_c * sin_ph_c)) -
                                (Jz.array() * sin_th_c)
                            ).array() * intg.array()
                        ).sum();
                            
                        // N_phi
                        // N_phi_intg = (-Jx * sin_ph) + (Jy * cos_ph)
                        // N_phi += np.sum(N_phi_intg * dS)
                        N_phi += (
                            (
                                (-Jx.array() * (sin_ph_c)) + 
                                (Jy.array() * (cos_ph_c))
                            ).array() * intg.array()
                        ).sum();

                        // # L_theta 
                        // L_theta_intg = (Mx * cos_th * cos_ph) + (My * cos_th * sin_ph) - (Mz * sin_th)
                        // L_theta += np.sum(L_theta_intg * dS)
                        L_theta += (
                            (
                                (Mx.array() * (cos_th_c * cos_ph_c)) + 
                                (My.array() * (cos_th_c * sin_ph_c)) -
                                (Mz.array() * sin_th_c)
                            ).array() * intg.array()
                        ).sum();

                        // # L_phi
                        // L_phi_intg = (-Mx * sin_ph) + (My * cos_ph)
                        // L_phi += np.sum(L_phi_intg * dS)
                        L_phi += (
                            (
                                (-Mx.array() * (sin_ph_c)) + 
                                (My.array() * (cos_ph_c))
                            ).array() * intg.array()
                        ).sum();

                    } // end faces loop 

                }  // end axis loop


                // pointer to thetapol result
                thetapol_p = (data_array + get_pointer_index_4(data_shape, {0, f, th, ph}));

                // pointer to phipol result
                phipol_p = (data_array + get_pointer_index_4(data_shape, {1, f, th, ph}));

                // contributions to N and L have been summed from all faces by this point. Calculate far-field
                // # electric field for thetapol and phipol
                // E_theta[i] = (-j * beta / (4 * np.pi)) * (L_phi + eta0 * N_theta)
                // E_phi[i] = (j * beta / (4 * np.pi)) * (L_theta - eta0 * N_phi)

            } // end frequency loop 

        } // end phi loop

    } // end theta loop

    return 0;
}
