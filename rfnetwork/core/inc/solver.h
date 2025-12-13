#ifndef SOLVER_H
#define SOLVER_H

#include <string>
#include <vector>
#include <thread>

#define N_FIELDS 18
#define N_COEFF 24

struct Field_Ex {
    float * ex_y;
    float * ex_z;
    float * ex;
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Field_Ey {
    float * ey_z;
    float * ey_x;
    float * ey;
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Field_Ez {
    float * ez_x;
    float * ez_y;
    float * ez;
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Field_Hx {
    float * hx_y;
    float * hx_z;
    float * hx;
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Field_Hy {
    float * hy_z;
    float * hy_x;
    float * hy;
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Field_Hz {
    float * hz_x;
    float * hz_y;
    float * hz;
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Coeff_Ex {
    float * Ca_ex_y;
    float * Ca_ex_z;
    float * Cb_ex_y;
    float * Cb_ex_z;
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Coeff_Ey {
    float * Ca_ey_z;
    float * Ca_ey_x;
    float * Cb_ey_z;
    float * Cb_ey_x;
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Coeff_Ez {
    float * Ca_ez_x;
    float * Ca_ez_y;
    float * Cb_ez_x;
    float * Cb_ez_y;
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Coeff_Hx {
    float * Da_hx_y;
    float * Da_hx_z;
    float * Db_hx_y;
    float * Db_hx_z;
};

struct Coeff_Hy {
    float * Da_hy_z;
    float * Da_hy_x;
    float * Db_hy_z;
    float * Db_hy_x;
};

struct Coeff_Hz {
    float * Da_hz_x;
    float * Da_hz_y;
    float * Db_hz_x;
    float * Db_hz_y;
};

struct Monitor {
    float * values;
    float * field;
    int axis;
    int position;
    int n_step;
    int N1;
    int N2;
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
    int N1N2;
};

struct Probe {
    float * values; // array of values for all time steps
    float * field_p; // pointer to field in grid
    int field_type;
    int x_cell; // cell index where the probe is located
    int yz_offset;
    int NyNz;
    bool is_source;
};

struct ThreadData {
    float * hy; // pointer to hz, hy field at the first x-index of the thread grid
    float * hz;
    float * ey; // pointer to the ez, ey field at the last x-index of the thread grid
    float * ez;
};

int solver_init_fields(PyObject * fields, PyObject * coefficients, int Nx, int Ny, int Nz);

int solver_init_monitors(PyObject * py_monitors, int Nt);

int solver_init_sources(PyObject * py_sources, int Nt);

int solver_init_probes(PyObject * py_probes, int Nt);

int solver_run(int Nt, int n_threads);

void solver_thread(int x_start, int x_stop, int Nt, int thread_idx);
void solver_controller(int Nt, int n_threads);

int solver_update_ex(int x_start, int x_stop);
int solver_update_ey(int x_start, int x_stop);
int solver_update_ez(int x_start, int x_stop);

int solver_update_hx(int x_start, int x_stop);
int solver_update_hy(int x_start, int x_stop);
int solver_update_hz(int x_start, int x_stop);

#endif /* SOLVER_H */