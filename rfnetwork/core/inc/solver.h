#ifndef SOLVER_H
#define SOLVER_H

#include <string>
#include <vector>
#include <thread>

#define N_FIELDS 18
#define N_COEFF 24

struct mbuffer_t
{
    float * base_addr; 
    float * next_addr;
    uint64_t available_size;
};

struct Field_Ex {
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Field_Ey {
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Field_Ez {
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Field_Hx {
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Field_Hy {
    int Nx;
    int Ny;
    int Nz;
    int NyNz;
};

struct Field_Hz {
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
    int Ny;
    int Nz;
    int NyNz;
};

struct Coeff_Ez {
    float * Ca_ez_x;
    float * Ca_ez_y;
    float * Cb_ez_x;
    float * Cb_ez_y;
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
    int field_type;
    int axis;
    int position;
    int n_step;
    int x_offset;
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
    float * field_s1_p; // pointer to first split field in grid
    float * field_s2_p; // pointer to second split field in grid
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

int solver_init_fields(PyObject * py_mem, PyObject * coefficients, int Nx, int Ny, int Nz);

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