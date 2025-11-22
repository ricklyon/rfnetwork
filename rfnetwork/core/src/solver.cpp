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


const char* FIELD_NAMES[]  = {
    "ex_y", "ex_z", "ex", 
    "ey_z", "ey_x", "ey", 
    "ez_x", "ez_y", "ez", 
    "hx_y", "hx_z", "hx", 
    "hy_z", "hy_x", "hy", 
    "hz_x", "hz_y", "hz", 
};

const char* COEFF_NAMES[] = {
    "Ca_ex_y", "Ca_ex_z", "Cb_ex_y", "Cb_ex_z",
    "Ca_ey_z", "Ca_ey_x", "Cb_ey_z", "Cb_ey_x",
    "Ca_ez_x", "Ca_ez_y", "Cb_ez_x", "Cb_ez_y",
    "Da_hx_y", "Da_hx_z", "Db_hx_y", "Db_hx_z",
    "Da_hy_z", "Da_hy_x", "Db_hy_z", "Db_hy_x",
    "Da_hz_x", "Da_hz_y", "Db_hz_x", "Db_hz_y",
};

int solver_coeff_index(const char* value)
{
    for (int i = 0; i < N_COEFF; i++) {
        if (strcmp(COEFF_NAMES[i], value) == 0)
            return i;
    }
    std::ostringstream oss;
    oss << "Coefficient value " << value << " not found.";
    throw std::runtime_error(oss.str());
    return -1;
}

int solver_field_index(const char* value)
{
    for (int i = 0; i < N_FIELDS; i++) {
        if (strcmp(FIELD_NAMES[i], value) == 0)
            return i;
    }
    std::ostringstream oss;
    oss << "Coefficient value " << value << " not found.";
    throw std::runtime_error(oss.str());
    return -1;
}

int solver_run(SolverConfig * sc)
{
    solver_efield(sc, 0, 2);
}

int solver_efield(SolverConfig * sc, int x_start, int x_stop)
{
    float * ex_y_p = sc->field[0];
    int Ny = sc->Ny;
    int Nz = sc->Nz;

    // get a single slice of the field along the x axis
    int x = x_start;
    MatrixFloatType ex_y (ex_y_p + (x * (Ny + 1) * (Nz + 1)), Ny + 1, Nz + 1);

    std::cout << "Ny " << Ny << "\n";
    ex_y.setConstant(3);

}
