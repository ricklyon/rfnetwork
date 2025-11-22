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
    solver_update_ex(sc, 0, sc->Nx);
    return 0;
}

// update Ex components, starting with the x-axis index x_start, and ending with x_stop (stop index is not inclusive)
int solver_update_ex(SolverConfig * sc, int x_start, int x_stop)
{
    int Nx = sc->Nx;
    int Ny = sc->Ny;
    int Nz = sc->Nz;

    float * ex_y_p = sc->field[solver_field_index("ex_y")];
    float * ex_z_p = sc->field[solver_field_index("ex_z")];
    float * ex_p = sc->field[solver_field_index("ex")];

    float * hz_p = sc->field[solver_field_index("hz")];
    float * hy_p = sc->field[solver_field_index("hz")];

    float * Cb_ex_y_p = sc->field[solver_coeff_index("Cb_ex_y")];
    float * Cb_ex_z_p = sc->field[solver_coeff_index("Cb_ex_z")];

    float * Ca_ex_y_p = sc->field[solver_coeff_index("Ca_ex_y")];
    float * Ca_ex_z_p = sc->field[solver_coeff_index("Ca_ex_z")];

    // get a single slice of the field along the x axis
    for (int x = x_start; x < x_stop; x++)
    {
        MatrixFloatType ex_y (ex_y_p + (x * (Ny + 1) * (Nz + 1)), Ny + 1, Nz + 1);
        MatrixFloatType ex_z (ex_z_p + (x * (Ny + 1) * (Nz + 1)), Ny + 1, Nz + 1);
        MatrixFloatType ex   (ex_p   + (x * (Ny + 1) * (Nz + 1)), Ny + 1, Nz + 1);

        MatrixFloatType hz (hz_p + (x * (Ny) * (Nz + 1)), Ny, Nz + 1);
        MatrixFloatType hy (hy_p + (x * (Ny + 1) * (Nz)), Ny + 1, Nz);

        MatrixFloatType Cb_ex_y (Cb_ex_y_p + (x * (Ny - 1) * (Nz - 1)), Ny - 1, Nz - 1);
        MatrixFloatType Cb_ex_z (Cb_ex_z_p + (x * (Ny - 1) * (Nz - 1)), Ny - 1, Nz - 1);

        MatrixFloatType Ca_ex_y (Ca_ex_y_p + (x * (Ny - 1) * (Nz - 1)), Ny - 1, Nz - 1);
        MatrixFloatType Ca_ex_z (Ca_ex_z_p + (x * (Ny - 1) * (Nz - 1)), Ny - 1, Nz - 1);
        
        // update ex_y
        // ex_y[:, 1:-1, 1:-1] = (Ca_ex_y * ex_y[:, 1:-1, 1:-1]) + ex_yd
        ex_y.block(1, 1, Ny -1, Nz -1) = (ex_y.block(1, 1, Ny -1, Nz -1).array() * Ca_ex_y.array()).matrix();

        // difference terms for hz along y
        // ex_yd = Cb_ex_y * np.diff(hz, axis=1)[:, :, 1:-1]
        ex_y.block(1, 1, Ny -1, Nz -1) += (
            Cb_ex_y.array() * (hz.bottomRows(Ny-1) - hz.topRows(Ny-1)).middleCols(1, Nz-2).array()
        ).matrix();

        // update ex_z
        // ex_z[:, 1:-1, 1:-1] = (Ca_ex_z * ex_z[:, 1:-1, 1:-1]) + ex_zd
        ex_z.block(1, 1, Ny -1, Nz -1) = (ex_z.block(1, 1, Ny -1, Nz -1).array() * Ca_ex_z.array()).matrix();

        // difference terms for hy along z
        // ex_zd = Cb_ex_z * np.diff(hy, axis=2)[:, 1:-1, :]
        ex_z.block(1, 1, Ny -1, Nz -1) += (
            Cb_ex_z.array() * (hy.rightCols(Nz-1) - hy.leftCols(Nz-1)).middleRows(1, Ny-2).array()
        ).matrix();

        // combine split components
        ex = ex_y + ex_z;
    }

    return 0;
}
