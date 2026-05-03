#ifndef SOLVER_H
#define SOLVER_H

#include <condition_variable>
#include <atomic>
#include <thread>
#include <complex>

struct _object;
typedef _object PyObject;

#define N_FIELDS 18
#define N_COEFF 24

#define MAX_MONITORS 50
#define MAX_PROBES 5000
#define MAX_THREADS 20


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

    float * Db_hx_y1;
    float * Db_hx_y2;
    
    float * Db_hx_z1;
    float * Db_hx_z2;
};

struct Coeff_Hy {
    float * Da_hy_z;
    float * Da_hy_x;

    float * Db_hy_z1;
    float * Db_hy_z2;

    float * Db_hy_x1;
    float * Db_hy_x2;
};

struct Coeff_Hz {
    float * Da_hz_x;
    float * Da_hz_y;

    float * Db_hz_x1;
    float * Db_hz_x2;

    float * Db_hz_y1;
    float * Db_hz_y2;
};

struct Monitor {
    char * values;
    std::complex<float> * dtft_phase;
    int n_phasors;
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

class SolverFDTD {

private:
    // coefficient values
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

    Monitor monitors[MAX_MONITORS];
    int n_monitors;

    Probe probes[MAX_PROBES];
    int n_probes;

    std::thread threads[MAX_THREADS];
    ThreadData thread_data[MAX_THREADS + 2];
    int n_threads;

    // conditional variables used by the thread controller and update threads for synchronization
    std::condition_variable cv;
    std::condition_variable cv_th;

    // shared variables protected by mutex
    std::mutex mutex;
    std::atomic<int> th_init{0};
    std::atomic<int> e_updates{0};
    std::atomic<int> h_updates{0};
    bool e_updates_done = false;
    bool h_updates_done = false;
    bool th_init_done = false;

    mbuffer_t m_pool{NULL, NULL, 0};

    float * mbuffer_allocate(uint64_t size);

    void mbuffer_init(float * base_addr, uint64_t size);
    void solver_thread(int x_start, int x_stop, int Nt, int thread_idx);

    int update_monitor(Monitor * mon, float * mon_field, int m_n, int x_start, int x_stop);
    int update_phasor_monitor(Monitor * mon, float * mon_field, int m_n, int x_start, int x_stop);

public:
    SolverFDTD();          // constructor
    int solver_init_fields(PyObject * py_mem, PyObject * coefficients, int Nx, int Ny, int Nz);
    int solver_init_monitors(PyObject * py_monitors, int Nt);
    int solver_init_probes(PyObject * py_probes, int Nt);

    int solver_run(int Nt, int n_threads, int update_interval);
    void solver_controller(int Nt, int n_threads, int update_interval);

    void solver_run_cu(int Nt);
};


#endif /* SOLVER_H */