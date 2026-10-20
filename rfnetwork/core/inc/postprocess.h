#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <string>
#include <vector>
#include <thread>
#include <complex>


int postprocess_nf2ff(
    PyObject * J_xyz, 
    PyObject * M_xyz, 
    PyObject * r_grid, 
    PyObject * w_grid, 
    PyObject * surf_pos, 
    PyObject * ff_data,
    int n_threads
);

int postprocess_nf2ff_thread(
    std::complex<float> * data_array,
    float * beta_arr,
    float * theta_arr,
    float * phi_arr,
    float surf_pos[3][2],
    float * r_grid_p[3][2],
    int JM_shape[3][4],
    int grid_shape[3][2],
    int * data_shape,
    std::complex<float> * ds_grid_p[3],
    std::complex<float> * M_xyz_p[3][2],
    std::complex<float> * J_xyz_p[3][2],
    std::complex<float> * working_grid_cmplx_p,
    float * working_grid_float_p,
    int theta_start,
    int theta_stop
);

int postprocess_nf2ff_flat(
    PyObject * J_xyz_py, 
    PyObject * M_xyz_py, 
    PyObject * cell_pos_py, 
    PyObject * cell_ds_py, 
    PyObject * ff_data_py,
    int n_threads
);

int postprocess_nf2ff_flat_thread(
    std::complex<float> * data_array,
    float * beta_arr,
    float * theta_arr,
    float * phi_arr,
    float * cell_pos_p,
    float * cell_ds_p,
    int * data_shape,
    std::complex<float> * M_xyz_p,
    std::complex<float> * J_xyz_p,
    std::complex<float> * working_grid_cmplx_p,
    float * working_grid_float_p,
    int theta_start,
    int theta_stop,
    int n_cells
);

#endif /* POSTPROCESS_H */