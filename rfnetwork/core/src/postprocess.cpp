
#define PY_SSIZE_T_CLEAN
#define _USE_MATH_DEFINES

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
#include <cmath>

#include "postprocess.h"

#include "Eigen/Dense"

using Eigen::MatrixXd;

typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixFloatType;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> MatrixFloatStride;
typedef Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixComplexType;

#define FFDATA_NDIM 4

// placement of far-field dims in data array
#define FF_POLARIZATION 0
#define FF_FREQUENCY 1
#define FF_THETA 2
#define FF_PHI 3

#define ETA0 376.730313

std::complex<float> * get_complex_array(PyObject* py_obj, int * shape, int ndim) 
{   
    PyArrayObject* array = (PyArrayObject*) py_obj;
    // get array shape
    npy_intp * npy_shape = PyArray_SHAPE(array);  

    std::ostringstream oss;

    if (ndim != PyArray_NDIM(array))
    {
        throw std::runtime_error("Invalid data array. Wrong number of dimensions.");
    }

    if (PyArray_TYPE(array) != NPY_CFLOAT)
    {
        throw std::runtime_error("Invalid data array. Must be complex float32 type.");
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

    return (std::complex<float> *) PyArray_DATA(array);
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

// Integrate J and M equivalent surface currents on a rectangular box to produce the far-field electric field values.
// See Section 6.8.2 in Balanis Advanced Engineering Electromagnetics 2nd Edition
int postprocess_nf2ff(
    PyObject * J_xyz_py, 
    PyObject * M_xyz_py, 
    PyObject * r_grid_py, 
    PyObject * ds_grid_py, 
    PyObject * surf_pos_py, 
    PyObject * ff_data_py
)
{
    npy_intp * npy_shape;

    // check that cell positions are given for the three cartesian axis
    if (PyList_Size(r_grid_py) != 3)
    {
        throw std::runtime_error("Invalid r_grid data array. Expected list of length 3.");
    }

    // check that cell areas are given for the three cartesian axis
    if (PyList_Size(ds_grid_py) != 3)
    {
        throw std::runtime_error("Invalid ds_grid data array. Expected list of length 3.");
    }

    // check surface positions
    if (PyList_Size(surf_pos_py) != 3)
    {
        throw std::runtime_error("Invalid surf_pos array. Expected list of length 3.");
    }

    // get the result data array where the E-fields will be held.
    PyObject* py_data = PyDict_GetItemString(ff_data_py, "data");
    PyArrayObject* py_data_arr = (PyArrayObject*) py_data;

    if (PyArray_NDIM(py_data_arr) != FFDATA_NDIM)
    {
        throw std::runtime_error("Invalid far-field data array. Expected four dimensions.");
    }

    // dimensions are polarization, frequency, theta, phi
    int data_shape[FFDATA_NDIM];
    npy_shape = PyArray_SHAPE(py_data_arr); 

    for (int i = 0; i < FFDATA_NDIM; ++i) { 
        data_shape[i] = (int) npy_shape[i];
    }

    // result data array
    std::complex<float> * data_array = get_complex_array(py_data, data_shape, FFDATA_NDIM);

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
    std::complex<float> * J_xyz_p[3][2];
    // array pointers for M currents, two per axis on each face
    std::complex<float> * M_xyz_p[3][2];
    // array pointers for cell positions on each axis
    float * r_grid_p[3][2];
    // array pointers for cell areas on each axis
    std::complex<float> * ds_grid_p[3];
    // surface positions, 2 values per axis
    float surf_pos[3][2];

    // get temporary working complex array
    PyObject* working_grid_py = PyDict_GetItemString(ff_data_py, "working_grid_cmplx");
    PyArrayObject* working_grid_array = (PyArrayObject*) working_grid_py;
    std::complex<float> * working_grid_cmplx_p = (std::complex<float> *) PyArray_DATA(working_grid_array);

    if (PyArray_TYPE(working_grid_array) != NPY_CFLOAT)
    {
        throw std::runtime_error("Invalid data array. Must be complex float type.");
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
        PyObject* ds_grid_axis = PyList_GetItem(ds_grid_py, axis);

        // get shape of the grid
        PyArrayObject* ds_grid_axis_arr = (PyArrayObject*) ds_grid_axis;
        npy_shape = PyArray_SHAPE(ds_grid_axis_arr);
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
        ds_grid_p[axis] = get_complex_array(ds_grid_axis, grid_shape[axis], 2);

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
    
    // variables for N and L auxilary fields
    std::complex<float> N_theta, N_phi, L_theta, L_phi;
    // pointers into the result data array for both polarizations
    std::complex<float> * thetapol_p;
    std::complex<float> * phipol_p;

    // variables for the current frequency, theta and phi values in the loops
    float beta, theta, phi;
    // working variables for cos(theta), sin(theta), etc...
    float cos_th, cos_ph, sin_th, sin_ph;
    // cos/sin terms cast as complex values
    std::complex<float> cos_th_c, cos_ph_c, sin_th_c, sin_ph_c;
    std::complex<float> beta_c;

    // 4 * PI as a complex number
    std::complex<float> pi4_c = (std::complex<float>) (4 * M_PI);
    // Impedance of free space as a complex number
    std::complex<float> eta0_c = (std::complex<float>) (ETA0);

    // variables for current uv values in the loop
    float u, v, w;
    // number of spatial grid points in a face
    int grid_size;

    // constant -1j, 1j and 0 as complex numbers
    std::complex<float> n1J = std::complex<float>(0, -1);
    std::complex<float> p1J = std::complex<float>(0, 1);
    std::complex<float> zero_c = std::complex<float>(0, 0);

    // loop over frequency
    for (int f = 0; f < data_shape[FF_FREQUENCY]; f++)
    {   
        beta = beta_arr[f];
        beta_c = (std::complex<float>) beta;
        
        std::complex<float> * Jx_p[3][2];
        std::complex<float> * Jy_p[3][2];
        std::complex<float> * Jz_p[3][2];

        std::complex<float> * Mx_p[3][2]; 
        std::complex<float> * My_p[3][2]; 
        std::complex<float> * Mz_p[3][2]; 


        // get pointers to each J and M grid at this frequency
        for (int axis = 0; axis < 3; axis++)
        {
            // each of the two faces on axis
            for (int s = 0; s < 2; s++)
            {
                Jx_p[axis][s] = (J_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {0, f, 0, 0});
                Jy_p[axis][s] = (J_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {1, f, 0, 0});
                Jz_p[axis][s] = (J_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {2, f, 0, 0});

                Mx_p[axis][s] = (M_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {0, f, 0, 0});
                My_p[axis][s] = (M_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {1, f, 0, 0});
                Mz_p[axis][s] = (M_xyz_p[axis][s]) + get_pointer_index_4(JM_shape[axis], {2, f, 0, 0});

            }
        }

        // loop over theta
        for (int th = 0; th < data_shape[FF_THETA]; th++)
        {
            theta = theta_arr[th];
            // loop over phi
            for (int ph = 0; ph < data_shape[FF_PHI]; ph++)
            {
                phi = phi_arr[ph];
                
                cos_ph = (float) std::cos((float) phi);
                cos_th = (float) std::cos((float) theta);
                sin_ph = (float) std::sin((float) phi);
                sin_th = (float) std::sin((float) theta);
                
                // covert to uvw
                u = sin_th * cos_ph;
                v = sin_th * sin_ph;
                w = cos_th;

                cos_ph_c = (std::complex<float>) cos_ph;
                cos_th_c = (std::complex<float>) cos_th;
                sin_ph_c = (std::complex<float>) sin_ph;
                sin_th_c = (std::complex<float>) sin_th;

                // reset N and L at each frequency/theta/phi, values from all faces are summed into a single value
                N_theta = zero_c;
                N_phi = zero_c;
                L_theta = zero_c;
                L_phi = zero_c;

                for (int axis = 0; axis < 3; axis++)
                {

                    grid_size = grid_shape[axis][0] * grid_shape[axis][1];

                    // each of the two faces on axis
                    for (int s = 0; s < 2; s++)
                    {
                        // Jx current grid
                        MatrixComplexType Jx (Jx_p[axis][s], grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // Jy current grid
                        MatrixComplexType Jy (Jy_p[axis][s], grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // Jz current grid
                        MatrixComplexType Jz (Jz_p[axis][s], grid_shape[axis][0], grid_shape[axis][1]);

                        // Mx current grid
                        MatrixComplexType Mx (Mx_p[axis][s], grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // My current grid
                        MatrixComplexType My (My_p[axis][s], grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // Mz current grid
                        MatrixComplexType Mz (Mz_p[axis][s], grid_shape[axis][0], grid_shape[axis][1]);

                        // cell positions on the surface
                        MatrixFloatType r_pos_0 (r_grid_p[axis][0], grid_shape[axis][0], grid_shape[axis][1]);
                        MatrixFloatType r_pos_1 (r_grid_p[axis][1], grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // cell widths
                        MatrixComplexType ds (ds_grid_p[axis], grid_shape[axis][0], grid_shape[axis][1]);

                        // phase term for integrand
                        MatrixFloatType r_dot (working_grid_float_p, grid_shape[axis][0], grid_shape[axis][1]);
                        
                        // (r_pos[0] * u + r_pos[1] * v + r_pos[2] * w))
                        if (axis == 0)
                        {
                            r_dot = ((surf_pos[axis][s] * u) + ((r_pos_0 * v).array() + (r_pos_1 * w).array()).array());
                        }
                        else if (axis == 1)
                        {
                            r_dot = ((surf_pos[axis][s] * v) + ((r_pos_0 * u).array() + (r_pos_1 * w).array()).array());
                        }
                        else
                        {
                            r_dot = ((surf_pos[axis][s] * w) + ((r_pos_0 * u).array() + (r_pos_1 * v).array()).array());
                        }
                        
                        // integrand term, same memory used for exponential phase term exp(1j * beta.item() * r_dot)
                        // and the differential dS.
                        MatrixComplexType intg (working_grid_cmplx_p, grid_shape[axis][0], grid_shape[axis][1]);

                        // phs_term = exp(1j * beta.item() * r_dot)
                        intg = exp((p1J * beta_c) * r_dot.array().cast<std::complex<float>>());
                        // dS = phs_term * d1 * d2
                        intg = intg.array() * (ds.array());

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
                *thetapol_p = ((n1J * beta_c / pi4_c) * (L_phi + (eta0_c * N_theta)));
                // E_phi[i] = (j * beta / (4 * np.pi)) * (L_theta - eta0 * N_phi)
                *phipol_p = ((p1J * beta_c / pi4_c) * (L_theta - (eta0_c * N_phi)));

            } // end phi loop

        } // end theta loop

    } // end frequency loop 

    return 0;
}
