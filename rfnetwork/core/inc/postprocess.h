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
    PyObject * ff_data
);

#endif /* POSTPROCESS_H */