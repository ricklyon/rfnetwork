
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
#include <condition_variable>
#include <atomic>

#include "solver.h"

#include "Eigen/Dense"

using Eigen::MatrixXd;

typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixFloatType;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> MatrixFloatStride;
typedef Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixComplexType;

typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> StrideType;

#define DATA_NDIM 3

#define EX 0
#define EY 1
#define EZ 2
#define HX 3
#define HY 4
#define HZ 5


long long get_milliseconds()
{

    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();

    return ms;
}

// ensures stdout is flushed so it works in jupyter notebooks
void print_stdout(std::stringstream& msg) {
    std::cout << msg.str();
    std::cout.flush();
}

// send progress bar to stdout
void print_progress(int n, int Nt)
{
    std::stringstream msg;
    int percent_completed = ((n + 1) * 100 / Nt);
    int n_dots = percent_completed / 5;
    int n_space = 20 - n_dots;

    msg << "\r|";
    for (int i = 0; i < n_dots; ++i) {
        msg << '.';
    }
    for (int i = 0; i < n_space; ++i) {
        msg << ' ';
    }

    msg << "| " << percent_completed << "%  ";
    print_stdout(msg);
}

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
            oss << "Invalid data array " << name << ". Expected shape on axis " << i << " of " << expected_shape[i];
            throw std::runtime_error(oss.str());
        }
    }

    return (float *) PyArray_DATA(array);
}

// get the array of a source from a python dictionary. Validate the shape
// matches Nt
float * get_source_array(PyObject * dict, int Nt) 
{
    PyObject* py_arr = PyDict_GetItemString(dict, "values");
    PyArrayObject* array = (PyArrayObject*) py_arr;
    // get array shape
    npy_intp * npy_shape = PyArray_SHAPE(array);  

    std::ostringstream oss;

    if (PyArray_NDIM(array) != 1)
    {
        throw std::runtime_error("Invalid source array. Wrong number of dimensions.");
    }

    if (PyArray_TYPE(array) != NPY_FLOAT)
    {
        throw std::runtime_error("Invalid data array. Must be float type.");
    }

    if (!(PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS))
    {
        oss << "Invalid source array. Must be row ordered (C-style)";
        throw std::runtime_error(oss.str());
    }

    if (((int) npy_shape[0]) < Nt)
    {
        oss << "Invalid data source array shape. Expected shape greater or equal to " << Nt;
        throw std::runtime_error(oss.str());
    }
    

    return (float *) PyArray_DATA(array);
}

std::complex<float> * get_complex_array(PyObject * dict, const char * name, int * shape, int ndim) 
{   

    PyObject* py_arr = PyDict_GetItemString(dict, name);
    PyArrayObject* array = (PyArrayObject*) py_arr;
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
            oss << "Invalid data array " << name << ". Expected shape on axis " << i << " of " << npy_shape[i];
            throw std::runtime_error(oss.str());
        }
    }

    return (std::complex<float> *) PyArray_DATA(array);
}


/*
Reserves a contigious section of memory of given size (size is number of floats) 
and returns the pointer to the memory section.
Returns NULL if requested size does not fit within the memory block.
*/
float * SolverFDTD::mbuffer_allocate(uint64_t size)
{   
    // allow only one thread at a time
    std::unique_lock<std::mutex> lock(mutex);

    // check that buffer has been initialized
    if (m_pool.base_addr == NULL)
    {
        throw std::runtime_error("Memory buffer not initialized.");
        return NULL;
    }

    // check that buffer has enough space
    if (size > m_pool.available_size)
    {
        throw std::runtime_error("Memory buffer has insufficent space.");
        return NULL;
    }

    float * addr = m_pool.next_addr;
    // increment buffer pointer for the next allocation
    m_pool.next_addr = m_pool.next_addr + size;
    // update available size
    m_pool.available_size -= size;

    return addr;
}

void SolverFDTD::mbuffer_init(float * base_addr, uint64_t size)
{
    m_pool.base_addr = base_addr;
    m_pool.next_addr = base_addr;
    m_pool.available_size = size;
}

SolverFDTD::SolverFDTD(){
    n_monitors = 0;
    n_probes = 0;
    n_threads = 0;
}

int SolverFDTD::solver_init_fields(PyObject * py_mem, PyObject * coefficients, int Nx_, int Ny_, int Nz_, int gpu)
{
    Nx = Nx_;
    Ny = Ny_;
    Nz = Nz_;

    int NyNz = Ny * Nz;

    // int Nxm1 = (gpu) ? Nx : Nx-1;
    int Nym1 = (gpu) ? Ny : Ny-1;
    int Nzm1 = (gpu) ? Nz : Nz-1;

    // int Nxp1 = (gpu) ? Nx : Nx+1;
    int Nyp1 = (gpu) ? Ny : Ny+1;
    int Nzp1 = (gpu) ? Nz : Nz+1;

    // error check memory buffer
    std::ostringstream oss;
    PyArrayObject * mem_array = (PyArrayObject *) py_mem;

    if (PyArray_TYPE(mem_array) != NPY_FLOAT)
    {
        throw std::runtime_error("Invalid data array. Must be float type.");
        return 1;
    }

    if (!(PyArray_FLAGS(mem_array) & NPY_ARRAY_C_CONTIGUOUS))
    {
        oss << "Invalid memory data array. Must be row ordered (C-style)";
        throw std::runtime_error(oss.str());
        return 1;
    }

    if ((int) PyArray_NDIM(mem_array) != 1)
    {
        throw std::runtime_error("Invalid memory buffer array. Must be 1 dimension.");
        return 1;
    }

    npy_intp * npy_shape = PyArray_SHAPE(mem_array); 

    // init memory buffer
    mbuffer_init((float *) PyArray_DATA(mem_array), (uint64_t) npy_shape[0]);

    // Cx
    Cx.Cb_ex_y = get_solver_array(coefficients, "Cb_ex_y", Nx, Nym1, Nzm1);
    Cx.Cb_ex_z = get_solver_array(coefficients, "Cb_ex_z", Nx, Nym1, Nzm1);
    Cx.Ca_ex_y = get_solver_array(coefficients, "Ca_ex_y", Nx, Nym1, Nzm1);
    Cx.Ca_ex_z = get_solver_array(coefficients, "Ca_ex_z", Nx, Nym1, Nzm1);

    // Cy
    Cy.Cb_ey_z = get_solver_array(coefficients, "Cb_ey_z", Nx, Ny, Nzm1);
    Cy.Cb_ey_x = get_solver_array(coefficients, "Cb_ey_x", Nx, Ny, Nzm1);
    Cy.Ca_ey_z = get_solver_array(coefficients, "Ca_ey_z", Nx, Ny, Nzm1);
    Cy.Ca_ey_x = get_solver_array(coefficients, "Ca_ey_x", Nx, Ny, Nzm1);
    // ensure coefficients at the end of the x-axis are zero to create a PEC boundary
    NyNz = Ny * Nzm1;
    memset(Cy.Cb_ey_z + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Cy.Cb_ey_x + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Cy.Ca_ey_z + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Cy.Ca_ey_x + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));

    // Cz
    Cz.Cb_ez_x = get_solver_array(coefficients, "Cb_ez_x", Nx, Nym1, Nz);
    Cz.Cb_ez_y = get_solver_array(coefficients, "Cb_ez_y", Nx, Nym1, Nz);
    Cz.Ca_ez_x = get_solver_array(coefficients, "Ca_ez_x", Nx, Nym1, Nz);
    Cz.Ca_ez_y = get_solver_array(coefficients, "Ca_ez_y", Nx, Nym1, Nz);
    // ensure coefficients at the end of the x-axis are zero to create a PEC boundary
    NyNz = Nym1 * Nz;
    memset(Cz.Cb_ez_x + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Cz.Cb_ez_y + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Cz.Ca_ez_x + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Cz.Ca_ez_y + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));

    // Dx
    Dx.Db_hx_y1 = get_solver_array(coefficients, "Db_hx_y1", Nx, Ny, Nz);
    Dx.Db_hx_y2 = get_solver_array(coefficients, "Db_hx_y2", Nx, Ny, Nz);
    Dx.Db_hx_z1 = get_solver_array(coefficients, "Db_hx_z1", Nx, Ny, Nz);
    Dx.Db_hx_z2 = get_solver_array(coefficients, "Db_hx_z2", Nx, Ny, Nz);
    
    Dx.Da_hx_y = get_solver_array(coefficients, "Da_hx_y", Nx, Ny, Nz);
    Dx.Da_hx_z = get_solver_array(coefficients, "Da_hx_z", Nx, Ny, Nz);
    // ensure coefficients at the end of the x-axis are zero to create a PEC boundary
    NyNz = Ny * Nz;
    memset(Dx.Db_hx_y1 + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Dx.Db_hx_y2 + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Dx.Db_hx_z1 + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Dx.Db_hx_z2 + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Dx.Da_hx_y + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));
    memset(Dx.Da_hx_z + ((Nx - 1) * NyNz), 0, NyNz * sizeof(float));

    // Dy
    Dy.Db_hy_z1 = get_solver_array(coefficients, "Db_hy_z1", Nx, Nyp1, Nz);
    Dy.Db_hy_z2 = get_solver_array(coefficients, "Db_hy_z2", Nx, Nyp1, Nz);
    Dy.Db_hy_x1 = get_solver_array(coefficients, "Db_hy_x1", Nx, Nyp1, Nz);
    Dy.Db_hy_x2 = get_solver_array(coefficients, "Db_hy_x2", Nx, Nyp1, Nz);

    Dy.Da_hy_z = get_solver_array(coefficients, "Da_hy_z", Nx, Nyp1, Nz);
    Dy.Da_hy_x = get_solver_array(coefficients, "Da_hy_x", Nx, Nyp1, Nz);

    // Dz
    Dz.Db_hz_x1 = get_solver_array(coefficients, "Db_hz_x1", Nx, Ny, Nzp1);
    Dz.Db_hz_x2 = get_solver_array(coefficients, "Db_hz_x2", Nx, Ny, Nzp1);
    Dz.Db_hz_y1 = get_solver_array(coefficients, "Db_hz_y1", Nx, Ny, Nzp1);
    Dz.Db_hz_y2 = get_solver_array(coefficients, "Db_hz_y2", Nx, Ny, Nzp1);

    Dz.Da_hz_x = get_solver_array(coefficients, "Da_hz_x", Nx, Ny, Nzp1);
    Dz.Da_hz_y = get_solver_array(coefficients, "Da_hz_y", Nx, Ny, Nzp1);

    return 0;
}

int SolverFDTD::solver_init_monitors(PyObject * py_monitors, int Nt)
{
    // initialize field monitors
    n_monitors = (int) PyList_Size(py_monitors);

    if (n_monitors > MAX_MONITORS)
    {
        throw std::runtime_error("Exceeded maximum number of monitors.");
    }

    // shapes of field components
    // the shapes of Ey, Ez, and Hx along x are incremented by 1 because the solver grid cells include these
    // components on the right edge of the cell, and the left most components are not included. 
    // int Nx[6] = {Ex.Nx, Ey.Nx+1, Ez.Nx+1, Hx.Nx+1, Hy.Nx, Hz.Nx};
    // int Ny[6] = {Ex.Ny, Ey.Ny, Ez.Ny, Hx.Ny, Hy.Ny, Hz.Ny};
    // int Nz[6] = {Ex.Nz, Ey.Nz, Ez.Nz, Hx.Nz, Hy.Nz, Hz.Nz};

    // allocated array length for each field type
    int f_Ny[6] = {Ny+1, Ny, Ny+1, Ny, Ny+1, Ny};
    int f_Nz[6] = {Nz+1, Nz+1, Nz, Nz, Nz, Nz+1};

    PyObject* py_mon;

    for (int m = 0; m < n_monitors; m++)
    {   
        py_mon = PyList_GetItem(py_monitors, m);

        // number of elements in time dimension
        int n_step = PyLong_AsLong(PyDict_GetItemString(py_mon, "n_step"));
        int Nm = (Nt / n_step) + 1;
        
        int axis = PyLong_AsLong(PyDict_GetItemString(py_mon, "axis"));
        int field = PyLong_AsLong(PyDict_GetItemString(py_mon, "field"));

        monitors[m].field_type = field;
        monitors[m].position = PyLong_AsLong(PyDict_GetItemString(py_mon, "position"));
        monitors[m].n_step = n_step;
        monitors[m].axis = axis;

        // monitor is on yz plane
        if (axis == 0)
        {
            // the thread grid for Hx, Ey, and Ez along x is offset by one compared to the global grid because the first cell
            // includes only the second component.
            if ((field == HX) || (field == EY) || (field == EZ))
            {
                monitors[m].position -= 1;
            }
            // each row (along y) skips by NyNz
            monitors[m].row_stride = f_Nz[field];
            // each column (along z) skips by 1
            monitors[m].col_stride = 1;
            monitors[m].yz_offset = 0;
            monitors[m].N1 = f_Ny[field];
            monitors[m].N2 = f_Nz[field];
        }
        // monitor is on xz plane
        else if (axis == 1)
        {
            // each row (along x) skips by NyNz
            monitors[m].row_stride = f_Ny[field] * f_Nz[field];
            // each column (along z) skips by 1
            monitors[m].col_stride = 1;
            monitors[m].yz_offset = (monitors[m].position) * f_Nz[field];
            monitors[m].N1 = Nx;
            monitors[m].N2 = f_Nz[field];
        }
        // monitor is on xy plane
        else
        {
            // each row (along x) skips by NyNz
            monitors[m].row_stride = f_Ny[field] * f_Nz[field];
            // each column (along y) skips by Nz
            monitors[m].col_stride = f_Nz[field];
            monitors[m].yz_offset = (monitors[m].position);
            monitors[m].N1 = Nx;
            monitors[m].N2 = f_Ny[field];
        }
        
        // monitor is frequency domain phasor if dtft phase is present in the dictionary
        if (PyDict_Contains(py_mon, PyUnicode_FromString("dtft_phase")))
        {   
            int n_frequencies = PyLong_AsLong(PyDict_GetItemString(py_mon, "n_frequencies"));
            int shape_dtft[2] = {Nm, n_frequencies};
            int shape_phasors[3] = {n_frequencies, monitors[m].N1, monitors[m].N2};
            monitors[m].dtft_phase = get_complex_array(py_mon, "dtft_phase", shape_dtft, 2);
            monitors[m].values = (char *) get_complex_array(py_mon, "values", shape_phasors, 3);
            monitors[m].n_phasors = n_frequencies;
        }
        else
        {
            monitors[m].values = (char *) get_solver_array(py_mon, "values", Nm, monitors[m].N1, monitors[m].N2);
            monitors[m].n_phasors = 0;
        }

        monitors[m].N1N2 = monitors[m].N1 * monitors[m].N2;
    }

    return 0;
}

int SolverFDTD::solver_init_probes(PyObject * py_probes, int Nt)
{
    // shapes of field components
    // int Nx[6] = {Ex.Nx, Ey.Nx, Ez.Nx, Hx.Nx, Hy.Nx, Hz.Nx};
    // int Ny[6] = {Ex.Ny, Ey.Ny, Ez.Ny, Hx.Ny, Hy.Ny, Hz.Ny};
    // int Nz[6] = {Ex.Nz, Ey.Nz, Ez.Nz, Hx.Nz, Hy.Nz, Hz.Nz};

    n_probes = (int) PyList_Size(py_probes);

    if (n_probes > MAX_PROBES)
    {
        throw std::runtime_error("Exceeded maximum number of probes.");
    }

    // get waveform data for each component where a source is present
    for (int s = 0; s < n_probes; s++) 
    {

        // verify each item in the list is a python dictionary
        PyObject * probe_dict = PyList_GetItem(py_probes, s);
        if (!PyDict_Check(probe_dict)) {
            throw std::runtime_error("Expected a list of probe dictionaries");
        }

        // get waveform data
        probes[s].values = get_source_array(probe_dict, Nt);

        // get component indices for ez
        PyObject* py_idx = PyDict_GetItemString(probe_dict, "idx");
        
        // cell that the field component belongs to
        probes[s].x_cell = (int) PyLong_AsLong(PyList_GetItem(py_idx, 0));
        probes[s].y_cell = (int) PyLong_AsLong(PyList_GetItem(py_idx, 1));
        probes[s].z_cell = (int) PyLong_AsLong(PyList_GetItem(py_idx, 2));

        int field_type = PyLong_AsLong(PyDict_GetItemString(probe_dict, "field"));
        probes[s].field_type = field_type;

        // if component's first index is at x=0 instead of x=0.5, the first usable component
        // is at index=1. The first thread does not capture the Ey, Ez, and Hx components at x=0. Each cell
        // contains the components in the middle and end of each yee grid cell along x.
        // offset x index for ey, ez, and hx components
        if ((field_type == EY) || (field_type == EZ) || (field_type == HX))
        {
            probes[s].x_cell -= 1;
        }
        // offset y index for ex, ez, and hy components
        if ((field_type == EX) || (field_type == EZ) || (field_type == HY))
        {
            probes[s].y_cell -= 1;
        }
        // offset z index for ex, ey and hz components
        if ((field_type == EX) || (field_type == EY) || (field_type == HZ))
        {
            probes[s].z_cell -= 1;
        }

        if ((probes[s].x_cell < 0) || (probes[s].y_cell < 0) || (probes[s].z_cell < 0))
        {
            throw std::runtime_error("Source cannot be placed on grid edge");
        }
        
        // flag probe if it provides source values
        probes[s].is_source = PyDict_GetItemString(probe_dict, "is_source");
    }

    return 0;
}

int SolverFDTD::solver_run(int Nt, int n_th, int update_interval)
{
    n_threads = n_th;
    th_init = 0;
    e_updates = 0;
    h_updates = 0;
    h_updates_done = false;
    e_updates_done = false;
    th_init_done = false;

    // error check number of threads
    if (n_threads < 0 || n_threads > MAX_THREADS)
    {
        throw std::runtime_error("Invalid number of threads.");
    }

    // number of x slices computed by each thread
    int n_batch = Nx / n_threads;
    // remainder of batch size
    int r_batch = Nx % n_threads;

    int x_start_th = 0;
    int x_stop_th = 0;

    // allocate dummy data for endpoints of thread grids. 
    thread_data[0].ez = mbuffer_allocate((Ny + 1) * Nz);
    thread_data[0].ey = mbuffer_allocate(Ny * (Nz + 1));
    thread_data[n_threads+1].hy = mbuffer_allocate((Ny + 1) * (Nz));
    thread_data[n_threads+1].hz = mbuffer_allocate((Ny) * (Nz + 1));

    // start controller thread
    std::thread control_th = std::thread(&SolverFDTD::solver_controller, this, Nt, n_threads, update_interval);

    // divide up the x slices into batches for each thread. The slices indicate the cells to compute
    // values for. Components that have an extra index on the x axis (ey, ez, and hx) are not updated on the edges and 
    // we only need to iterate over the number of cells. 
    // (hy is actually updated, but is always non-zero since ey and ez are not updated.)
    for (int t = 0; t < n_threads; t++)
    {
        x_start_th = x_stop_th;
        
        // add 1 to the batch size until the remainder is removed
        if (r_batch > 0)
        {
            x_stop_th =  x_start_th + n_batch + 1;
            r_batch -= 1;
        }
        else
        {
            x_stop_th =  x_start_th + n_batch;
        }
        // offset the thread index by one so the thread accesses the dummy endpoint data if it's the
        // first or last thread.
        threads[t] = std::thread(&SolverFDTD::solver_thread, this, x_start_th, x_stop_th, Nt, t+1);
    }

    // wait for all threads to complete
    for (int t = 0; t < n_threads; t++)
    {
        threads[t].join();
    }

    control_th.join();

    return 0;
}



// thread responsible for synchronizing field update threads
void SolverFDTD::solver_controller(int Nt, int n_threads, int update_interval)
{
    std::stringstream msg;

    // wait until all threads have signaled they are done with memory allocation
    {
        // get lock on mutex while shared variables are modified
        std::unique_lock<std::mutex> lock(mutex);

        // wait until all threads are done with memory allocation
        cv.wait(lock, [n_threads, this] { return th_init.load() == n_threads; });
        th_init_done = true;
        cv_th.notify_all();
    }

    if (update_interval)
    {
        print_progress(0, Nt);
    }
    
    for (int n = 0; n < Nt; n++)
    {
        // wait until all threads have signaled they are done with e field updates
        {
            // get lock on mutex while shared variables are modified
            std::unique_lock<std::mutex> lock(mutex);
            h_updates = 0;
            h_updates_done = true;
            e_updates_done = false;
            // send notification to threads that H-field updates are done. On the first iteration, the threads
            // will ignore this because they are first waiting that e-field updates are done.
            cv_th.notify_all();
            cv.wait(lock, [n_threads, this] { return e_updates.load() == n_threads; });
        }


        // reset e update counter and signal to threads that they can move on to h updates, wait until h updates 
        // are completed on all threads
        {
            // get lock on mutex while shared variables are modified
            std::unique_lock<std::mutex> lock(mutex);
            e_updates = 0;
            e_updates_done = true;
            h_updates_done = false;
            // send notification to threads that E-field updates are done.
            cv_th.notify_all();
            cv.wait(lock, [n_threads, this] { return h_updates.load() == n_threads; });
        }

        // write update
        if ((update_interval) && ((n % update_interval) == 0))
        {
            print_progress(n, Nt);
        }
    }

    // send a final notification that h field updates are complete
    {
        // get lock on mutex while shared variables are modified
        std::unique_lock<std::mutex> lock(mutex);
        h_updates = 0;
        h_updates_done = true;
        e_updates_done = false;
        cv_th.notify_all();
    }

    if (update_interval > 0) 
    {
        print_progress(Nt, Nt);
    }
}

void SolverFDTD::solver_thread(int x_start, int x_stop, int Nt, int thread_idx)
{
    // each grid cell contains the ex, ey, hz components in the middle of the x axis of the cell, and the
    // ey, ez, hx components on the RIGHT side of the cell. The first cell does not update the ey, ez, hx components
    // on the left

    // x_stop is non-inclusive

    std::stringstream msg;

    int x_offset;
    // number of updated field components
    int Nx = x_stop - x_start;
    // int NyNz = Ny * Nz;

    // allocate memory for this thread's grid.
    // only Nx components are created for all fields, the Ey, Ez, and Hx components that have one extra 
    // component do not track the fields on the first index. They are not updated as they are on the edge of the grid
    // and always remain at zero.

    // allow only one thread at a time
    // std::unique_lock<std::mutex> lock(mutex);

    // int Nxp1 = Nx + 1;
    int Nyp1 = Ny + 1;
    int Nzp1 = Nz + 1;

    // int Nxm1 = Nx - 1;
    int Nym1 = Ny - 1;
    int Nzm1 = Nz - 1;

    int ex_NyNz = (Nyp1) * (Nzp1);
    int ey_NyNz = (Ny) * (Nzp1);
    int ez_NyNz = (Nyp1) * (Nz);

    int hx_NyNz = (Ny) * (Nz);
    int hy_NyNz = (Nyp1) * (Nz);
    int hz_NyNz = (Ny) * (Nzp1);

    int cx_NyNz = Nym1 * Nzm1;
    int cy_NyNz = Ny * Nzm1;
    int cz_NyNz = Nym1 * Nz;

    int dx_NyNz = Ny * Nz;
    int dy_NyNz = Nyp1 * Nz;
    int dz_NyNz = Ny * Nzp1;

    // include extra component at the lower edge of y and z axis.
    // the first component along x of Ey, Ez and Hx is not included
    float * p_ex_y = mbuffer_allocate(Nx * ex_NyNz); // 
    float * p_ex_z = mbuffer_allocate(Nx * ex_NyNz); // 
    float * p_ex   = mbuffer_allocate(Nx * ex_NyNz); // 

    float * p_ey_z = mbuffer_allocate(Nx * ey_NyNz); //  
    float * p_ey_x = mbuffer_allocate(Nx * ey_NyNz); //  
    float * p_ey   = mbuffer_allocate(Nx * ey_NyNz); // 

    float * p_ez_x = mbuffer_allocate(Nx * ez_NyNz); //  
    float * p_ez_y = mbuffer_allocate(Nx * ez_NyNz); //  
    float * p_ez   = mbuffer_allocate(Nx * ez_NyNz); // 

    // don't include h-components at the edge of the grid
    float * p_hx_y = mbuffer_allocate(Nx * hx_NyNz); //  
    float * p_hx_z = mbuffer_allocate(Nx * hx_NyNz); //  
    float * p_hx   = mbuffer_allocate(Nx * hx_NyNz); //  

    float * p_hy_z = mbuffer_allocate(Nx * hy_NyNz); //  
    float * p_hy_x = mbuffer_allocate(Nx * hy_NyNz); //  
    float * p_hy   = mbuffer_allocate(Nx * hy_NyNz); //  

    float * p_hz_x = mbuffer_allocate(Nx * hz_NyNz); //  
    float * p_hz_y = mbuffer_allocate(Nx * hz_NyNz); //  
    float * p_hz   = mbuffer_allocate(Nx * hz_NyNz); //  

    // populate thread data. Hy and Hz at the beginning of the x-block are used by the previous thread to update
    // the E fields at the edge. Ey and Ez are used by the next thread to update the H fields.
    thread_data[thread_idx].hy = p_hy;
    thread_data[thread_idx].hz = p_hz;
    thread_data[thread_idx].ey = p_ey + ((Nx - 1) * ey_NyNz);
    thread_data[thread_idx].ez = p_ez + ((Nx - 1) * ez_NyNz);

    // temporary variables
    float * p_hz_1; // points to the hz components in the next thread grid
    float * p_hy_1;
    float * p_ey_0; // points to the ey components in the previous thread grid
    float * p_ez_0; // points to the ez components in the previous thread grid
    float * mon_field; // points to a field that is being monitored

    // get all probes indices that fall within this thread's grid
    std:: vector<Probe*> e_probes;
    std:: vector<Probe*> h_probes;
    Probe * p;
    float * fields_base[6]     = {p_ex, p_ey, p_ez, p_hx, p_hy, p_hz};
    float * fields_sp1_base[6] = {p_ex_y, p_ey_z, p_ez_x, p_hx_y, p_hy_z, p_hz_x};
    float * fields_sp2_base[6] = {p_ex_z, p_ey_x, p_ez_y, p_hx_z, p_hy_x, p_hz_y};

    int px, py, pz;
    // int ftype;
    for (int i = 0; i < n_probes; i++)
    {   
        p = &(probes[i]);
        // if probe is inside the grid for this thread
        if (((p->x_cell) >= x_start) && ((p->x_cell) < x_stop))
        {
            px = p->x_cell;
            py = p->y_cell;
            pz = p->z_cell;

            // translate probe cell to an integer offset into the field array
            // ex
            if ((p->field_type) == EX)
            {
                x_offset = ((px - x_start) * ex_NyNz) + ((py + 1) * Nzp1) + (pz + 1);
                e_probes.push_back(p);
            }
            // ey
            else if ((p->field_type) == EY)
            {
                x_offset = ((px - x_start) * ey_NyNz) + ((py) * Nzp1) + (pz + 1);
                e_probes.push_back(p);
            }
            // ez
            else if ((p->field_type) == EZ)
            {
                x_offset = ((px - x_start) * ez_NyNz) + ((py + 1) * Nz) + (pz);
                e_probes.push_back(p);
            }
            // add h-field probe
            // hx
            else if ((p->field_type) == HX)
            {
                x_offset = ((px - x_start) * hx_NyNz) + ((py) * Nz) + (pz);
                h_probes.push_back(p);
            }
            else if ((p->field_type) == HY)
            {
                x_offset = ((px - x_start) * hy_NyNz) + ((py + 1) * Nz) + (pz);
                h_probes.push_back(p);
            }
            else if ((p->field_type) == HZ)
            {
                x_offset = ((px - x_start) * hz_NyNz) + ((py) * Nzp1) + (pz + 1);
                h_probes.push_back(p);
            }
            // compute the pointer for the probe in the grid. x_index needs to account for the starting position of 
            // the grid.
            
            p->field_p = (fields_base[p->field_type]) + x_offset;
            p->field_s1_p = (fields_sp1_base[p->field_type]) + x_offset;
            p->field_s2_p = (fields_sp2_base[p->field_type]) + x_offset;

        }
    }


    // msg.str("");
    // msg.clear();
    // msg << "Start Thread " << thread_idx << "... \n";
    // std::cout << msg.str();

    {
        // lock the mutex while updating shared variable, also ensures that only one thread sends a notification
        // to the controller at a time, preventing missed notifications.
        std::unique_lock<std::mutex> lock(mutex);
        ++th_init;

        cv.notify_all(); 
        // wait for all threads to reach this point before moving on to h-field updates.
        cv_th.wait(lock, [this] { return th_init_done; });
    }

    // main time stepping loop
    for (int n = 0; n < Nt; n++)
    {
        // operate on a single slice of the field on the x axis
        for (int x = 0; x < Nx; x++)
        {   
            x_offset = x * ex_NyNz;
            MatrixFloatType ex_y (p_ex_y + x_offset, Nyp1, Nzp1);
            MatrixFloatType ex_z (p_ex_z + x_offset, Nyp1, Nzp1);
            MatrixFloatType ex   (p_ex   + x_offset, Nyp1, Nzp1);
            
            x_offset = x * ey_NyNz;
            MatrixFloatType ey_z (p_ey_z + x_offset, Ny, Nzp1);
            MatrixFloatType ey_x (p_ey_x + x_offset, Ny, Nzp1);
            MatrixFloatType ey   (p_ey   + x_offset, Ny, Nzp1);
            
            x_offset = x * ez_NyNz;
            MatrixFloatType ez_x (p_ez_x + x_offset, Nyp1, Nz);
            MatrixFloatType ez_y (p_ez_y + x_offset, Nyp1, Nz);
            MatrixFloatType ez   (p_ez   + x_offset, Nyp1, Nz);

            // h-fields
            MatrixFloatType hx   (p_hx   + (x * hx_NyNz), Ny, Nz);
            MatrixFloatType hy   (p_hy   + (x * hy_NyNz), Nyp1, Nz);
            MatrixFloatType hz   (p_hz   + (x * hz_NyNz), Ny, Nzp1);

            // ex coefficients
            x_offset = (x_start + x) * (cx_NyNz);
            MatrixFloatType Cb_ex_y (Cx.Cb_ex_y + x_offset, Nym1, Nzm1);
            MatrixFloatType Cb_ex_z (Cx.Cb_ex_z + x_offset, Nym1, Nzm1);

            MatrixFloatType Ca_ex_y (Cx.Ca_ex_y + x_offset, Nym1, Nzm1);
            MatrixFloatType Ca_ex_z (Cx.Ca_ex_z + x_offset, Nym1, Nzm1);

            // ey coefficients
            x_offset = (x_start + x) * (cy_NyNz);
            MatrixFloatType Cb_ey_z (Cy.Cb_ey_z + x_offset, Ny, Nzm1);
            MatrixFloatType Cb_ey_x (Cy.Cb_ey_x + x_offset, Ny, Nzm1);

            MatrixFloatType Ca_ey_z (Cy.Ca_ey_z + x_offset, Ny, Nzm1);
            MatrixFloatType Ca_ey_x (Cy.Ca_ey_x + x_offset, Ny, Nzm1);
            
            // ez coefficients
            x_offset = (x_start + x) * (cz_NyNz);
            MatrixFloatType Cb_ez_x (Cz.Cb_ez_x + x_offset, Nym1, Nz);
            MatrixFloatType Cb_ez_y (Cz.Cb_ez_y + x_offset, Nym1, Nz);

            MatrixFloatType Ca_ez_x (Cz.Ca_ez_x + x_offset, Nym1, Nz);
            MatrixFloatType Ca_ez_y (Cz.Ca_ez_y + x_offset, Nym1, Nz);


            // next cell components. Dummy cells provide all zero components for the threads at the end points
            // of the grid
            p_hz_1 = (x < (Nx - 1)) ? p_hz + ((x + 1) * hz_NyNz) : (thread_data[thread_idx + 1]).hz;
            p_hy_1 = (x < (Nx - 1)) ? p_hy + ((x + 1) * hy_NyNz) : (thread_data[thread_idx + 1]).hy;
            MatrixFloatType hz_1 (p_hz_1, Ny, Nzp1);
            MatrixFloatType hy_1 (p_hy_1, Nyp1, Nz);

            // ----------------- update ex -------------------------- //
            // ex_y update
            // ex_yd = Cb_ex_y * np.diff(hz, axis=1)[:, :, 1:-1]
            // ex_y[:, 1:-1, 1:-1] = (Ca_ex_y * ex_y[:, 1:-1, 1:-1]) + ex_yd
            ex_y.block(1, 1, Nym1, Nzm1) = Ca_ex_y.cwiseProduct(ex_y.block(1, 1, Nym1, Nzm1)) + (
                Cb_ex_y.cwiseProduct((hz.bottomRows(Nym1) - hz.topRows(Nym1)).block(0, 1, Nym1, Nzm1))
            );

            // ex_z update
            // ex_zd = Cb_ex_z * np.diff(hy, axis=2)[:, 1:-1, :]
            // ex_z[:, 1:-1, 1:-1] = (Ca_ex_z * ex_z[:, 1:-1, 1:-1]) + ex_zd
            ex_z.block(1, 1, Nym1, Nzm1) = Ca_ex_z.cwiseProduct(ex_z.block(1, 1, Nym1, Nzm1) ) + (
                Cb_ex_z.cwiseProduct((hy.rightCols(Nzm1) - hy.leftCols(Nzm1)).block(1, 0, Nym1, Nzm1))
            );

            // ----------------- update ey -------------------------- //
            // ey_z update
            // ey_zd = Cb_ey_z * np.diff(hx, axis=2)[1:-1, :, :]
            // ey_z[1:-1, :, 1:-1] = (Ca_ey_z * ey_z[1:-1, :, 1:-1]) + ey_zd
            ey_z.block(0, 1, Ny, Nzm1) = Ca_ey_z.cwiseProduct(ey_z.block(0, 1, Ny, Nzm1)) + (
                Cb_ey_z.cwiseProduct(hx.rightCols(Nzm1) - hx.leftCols(Nzm1))
            );

            // ey_x update
            // ey_xd = Cb_ey_x * np.diff(hz, axis=0)[:, :, 1:-1]
            // ey_x[1:-1, :, 1:-1] = (Ca_ey_x * ey_x[1:-1, :, 1:-1]) + ey_xd
            ey_x.block(0, 1, Ny, Nzm1)  = Ca_ey_x.cwiseProduct(ey_x.block(0, 1, Ny, Nzm1)) + (
                Cb_ey_x.cwiseProduct((hz_1 - hz).block(0, 1, Ny, Nzm1))
            );
            
            // ----------------- update ez -------------------------- //
            // ez_x update
            // ez_xd = Cb_ez_x * np.diff(hy, axis=0)[:, 1:-1, :]
            // ez_x[1:-1, 1:-1, :] = (Ca_ez_x * ez_x[1:-1, 1:-1, :]) + ez_xd
            // get hy components on either side of x-slice, the hz component below ez is in the same cell,
            ez_x.block(1, 0, Nym1, Nz)  = Ca_ez_x.cwiseProduct(ez_x.block(1, 0, Nym1, Nz) ) + (
                Cb_ez_x.cwiseProduct((hy_1 - hy).block(1, 0, Nym1, Nz))
            );

            // update ez_y
            // ez_yd = Cb_ez_y * np.diff(hx, axis=1)[1:-1, :, :]
            // ez_y[1:-1, 1:-1, :] = (Ca_ez_y * ez_y[1:-1, 1:-1, :]) + ez_yd
            ez_y.block(1, 0, Nym1, Nz)  = Ca_ez_y.cwiseProduct(ez_y.block(1, 0, Nym1, Nz) ) + (
                Cb_ez_y.cwiseProduct(hx.bottomRows(Nym1) - hx.topRows(Nym1))
            );

            // h components have an extra component past the edge of the grid
            // make sure the D and C coefficients are set to zero at the edges because they will be updated

            // e components include the first index so the h-field can be updated

            // combine split components
            ex = ex_y + ex_z;
            ey = ey_z + ey_x;
            ez = ez_x + ez_y;
        }

        // update e-probe values
        for (Probe * p : e_probes) {
            // apply soft source
            if (p->is_source)
            {
                *(p->field_s1_p) = *(p->field_s1_p) + (p->values)[n];
                *(p->field_s2_p) = *(p->field_s2_p) + (p->values)[n];
                *(p->field_p) = *(p->field_s1_p) + *(p->field_s2_p);
            }

            // update probes
            // if this is a source, the values array is replaced with the resulting total voltage after it is used for
            // each time step.
            (p->values)[n] = *(p->field_p);
        }


        // signal to controller that all E-updates are done
        {
            // lock the mutex while updating shared variable, also ensures that only one thread sends a notification
            // to the controller at a time, preventing missed notifications.
            std::unique_lock<std::mutex> lock(mutex);
            ++e_updates;

            cv.notify_all(); 
            // wait for all threads to reach this point before moving on to h-field updates.
            cv_th.wait(lock, [this] { return e_updates_done; });
        }

        // h-updates
        for (int x = 0; x < Nx; x++)
        {   
            x_offset = x * hx_NyNz;
            MatrixFloatType hx_y (p_hx_y + x_offset, Ny, Nz);
            MatrixFloatType hx_z (p_hx_z + x_offset, Ny, Nz);
            MatrixFloatType hx   (p_hx   + x_offset, Ny, Nz);

            x_offset = x * hy_NyNz;
            MatrixFloatType hy_z (p_hy_z + x_offset, Nyp1, Nz);
            MatrixFloatType hy_x (p_hy_x + x_offset, Nyp1, Nz);
            MatrixFloatType hy   (p_hy   + x_offset, Nyp1, Nz);

            x_offset = x * hz_NyNz;
            MatrixFloatType hz_x (p_hz_x + x_offset, Ny, Nzp1);
            MatrixFloatType hz_y (p_hz_y + x_offset, Ny, Nzp1);
            MatrixFloatType hz   (p_hz   + x_offset, Ny, Nzp1);

            // e-fields 
            MatrixFloatType ex   (p_ex   + (x * ex_NyNz), Nyp1, Nzp1);
            MatrixFloatType ey   (p_ey   + (x * ey_NyNz), Ny, Nzp1);
            MatrixFloatType ez   (p_ez   + (x * ez_NyNz), Nyp1, Nz);

            // hx coefficients
            x_offset = (x_start + x) * dx_NyNz;
            MatrixFloatType Db_hx_y1 (Dx.Db_hx_y1 + x_offset, Ny, Nz);
            MatrixFloatType Db_hx_y2 (Dx.Db_hx_y2 + x_offset, Ny, Nz);
            MatrixFloatType Db_hx_z1 (Dx.Db_hx_z1 + x_offset, Ny, Nz);
            MatrixFloatType Db_hx_z2 (Dx.Db_hx_z2 + x_offset, Ny, Nz);

            MatrixFloatType Da_hx_y (Dx.Da_hx_y + x_offset, Ny, Nz);
            MatrixFloatType Da_hx_z (Dx.Da_hx_z + x_offset, Ny, Nz);

            // hy coefficients
            x_offset = (x_start + x) * dy_NyNz;
            MatrixFloatType Db_hy_z1 (Dy.Db_hy_z1 + x_offset, Nyp1, Nz);
            MatrixFloatType Db_hy_z2 (Dy.Db_hy_z2 + x_offset, Nyp1, Nz);
            MatrixFloatType Db_hy_x1 (Dy.Db_hy_x1 + x_offset, Nyp1, Nz);
            MatrixFloatType Db_hy_x2 (Dy.Db_hy_x2 + x_offset, Nyp1, Nz);

            MatrixFloatType Da_hy_z (Dy.Da_hy_z + x_offset, Nyp1, Nz);
            MatrixFloatType Da_hy_x (Dy.Da_hy_x + x_offset, Nyp1, Nz);

            // hz coefficients
            x_offset = (x_start + x) * dz_NyNz;
            MatrixFloatType Db_hz_x1 (Dz.Db_hz_x1 + x_offset, Ny, Nzp1);
            MatrixFloatType Db_hz_x2 (Dz.Db_hz_x2 + x_offset, Ny, Nzp1);
            MatrixFloatType Db_hz_y1 (Dz.Db_hz_y1 + x_offset, Ny, Nzp1);
            MatrixFloatType Db_hz_y2 (Dz.Db_hz_y2 + x_offset, Ny, Nzp1);

            MatrixFloatType Da_hz_x (Dz.Da_hz_x + x_offset, Ny, Nzp1);
            MatrixFloatType Da_hz_y (Dz.Da_hz_y + x_offset, Ny, Nzp1);

            // previous cell components. Dummy cells provide all zero components for the threads at the end points
            // of the grid
            p_ey_0 = (x > 0) ? p_ey + ((x - 1) * ey_NyNz) : (thread_data[thread_idx - 1]).ey;
            p_ez_0 = (x > 0) ? p_ez + ((x - 1) * ez_NyNz) : (thread_data[thread_idx - 1]).ez;
            MatrixFloatType ey_0 (p_ey_0, Ny, Nzp1);
            MatrixFloatType ez_0 (p_ez_0, Nyp1, Nz);

            // ----------------- update hx -------------------------- //
            // hx_y update
            // hx_yd = Db_hx_y * np.diff(ez, axis=1)
            // hx_y = Da_hx_y * hx_y + hx_yd
            hx_y = Da_hx_y.cwiseProduct(hx_y) + (
                Db_hx_y2.cwiseProduct(ez.bottomRows(Ny)) - Db_hx_y1.cwiseProduct(ez.topRows(Ny))
            );

            // hx_z update
            // hx_zd = Db_hx_z * np.diff(ey, axis=2)
            // hx_z = Da_hx_z * hx_z + hx_zd
            hx_z = Da_hx_z.cwiseProduct(hx_z) + (
                Db_hx_z2.cwiseProduct(ey.rightCols(Nz)) - Db_hx_z1.cwiseProduct(ey.leftCols(Nz))
            );
            
            // ----------------- update hy -------------------------- //
            // hy_z update
            // hy_zd = Db_hy_z * np.diff(ex, axis=2)
            // hy_z = Da_hy_z * hy_z + hy_zd
            hy_z = Da_hy_z.cwiseProduct(hy_z) + (
                Db_hy_z2.cwiseProduct(ex.rightCols(Nz)) - Db_hy_z1.cwiseProduct(ex.leftCols(Nz))
            );

            // update hy_x
            // hy_xd = Db_hy_x * np.diff(ez, axis=0)
            // hy_x = Da_hy_x * hy_x + hy_xd
            hy_x = Da_hy_x.cwiseProduct(hy_x) + (
                Db_hy_x2.cwiseProduct(ez) - Db_hy_x1.cwiseProduct(ez_0)
            );

            // ----------------- update hz -------------------------- //
            // hz_x update
            // hz_xd = Db_hz_x * np.diff(ey, axis=0) 
            // hz_x = Da_hz_x * hz_x + hz_xd
            hz_x = Da_hz_x.cwiseProduct(hz_x) + (
                Db_hz_x2.cwiseProduct(ey) - Db_hz_x1.cwiseProduct(ey_0)
            );

            // update hz_y
            // hz_yd = Db_hz_y * np.diff(ex, axis=1)
            // hz_y = Da_hz_y * hz_y + hz_yd
            hz_y = Da_hz_y.cwiseProduct(hz_y) + (
                Db_hz_y2.cwiseProduct(ex.bottomRows(Ny)) - Db_hz_y1.cwiseProduct(ex.topRows(Ny))
            );

            // combine split components
            hx = hx_y + hx_z;
            hy = hy_z + hy_x;
            hz = hz_x + hz_y;

        }

        // update h-probe values
        for (Probe * p : h_probes) 
        {
            // apply soft source
            if (p->is_source)
            {
                *(p->field_s1_p) = *(p->field_s1_p) + (p->values)[n];
                *(p->field_s2_p) = *(p->field_s2_p) + (p->values)[n];
                *(p->field_p) = *(p->field_s1_p) + *(p->field_s2_p);
            }
            // put the resulting total voltage in the source_values array once the value is used for this
            // time step.
            (p->values)[n] = *(p->field_p);
        }

        // wait for all threads to complete before moving on to e fields
        {
            // lock the mutex
            std::unique_lock<std::mutex> lock(mutex);
            ++h_updates;

            // notify controller that h_updates has been changed
            cv.notify_all(); 
            cv_th.wait(lock, [this] { return h_updates_done; });
        }

        // update monitors.
        for (int m = 0; m < n_monitors; m++)
        {
            Monitor mon = monitors[m];
            // pointer to field in grid
            mon_field = fields_base[mon.field_type];

            // only update if at the time step interval of the monitor
            if ((n > 0) && ((n % (mon.n_step)) != 0))
            {
                continue;
            }

            // only update x slice monitor if it is within bounds of the thread.
            // position has already been corrected in init_monitors to account for the fact that first
            // grid cell in the thread grid starts at x=1 in the global grid for the Ez, Hx, and Ey components.
            if ((mon.axis == 0) && (((mon.position) < x_start) || ((mon.position) >= x_stop)))
            {
                continue;
            }
            
            int m_n = n / mon.n_step;

            if (mon.n_phasors)
            {
                update_phasor_monitor(&mon, mon_field, m_n, x_start, x_stop);
            }

            else
            {
                update_monitor(&mon, mon_field, m_n, x_start, x_stop);
            }
        }

    } // end time stepping

}


int SolverFDTD::update_monitor(Monitor * mon, float * mon_field, int m_n, int x_start, int x_stop)
{

    MatrixFloatType m_values (((float *) (mon->values)) + (m_n * (mon->N1N2)), mon->N1, mon->N2);

    if (mon->axis == 0)
    {
        MatrixFloatType m_field (mon_field + ((mon->position - x_start) * (mon->N1N2)), mon->N1, mon->N2);
        m_values = m_field;
    }

    else
    {
        MatrixFloatStride m_field (
            mon_field + mon->yz_offset, 
            (x_stop - x_start),
            mon->N2,
            StrideType(mon->row_stride, mon->col_stride)
        );
        m_values.block(x_start, 0, x_stop - x_start, mon->N2) = m_field;
    }

    return 0;
}


int SolverFDTD::update_phasor_monitor(Monitor * mon, float * mon_field, int m_n, int x_start, int x_stop)
{

    for (int f = 0; f < mon->n_phasors; f++)
    {
        std::complex<float> * m_values_p = ((std::complex<float> *) (mon->values)) + (f * (mon->N1N2));

        // pointer to matrix that holds the monitor results
        MatrixComplexType m_values (m_values_p, mon->N1, mon->N2);
        // get dtft exponential phase term at time step and frequency
        std::complex<float> * dtft_phase_p = ((std::complex<float> *) (mon->dtft_phase)) + ((m_n * (mon->n_phasors)) + f);
        std::complex<float> dtft_phase = *dtft_phase_p;

        if (mon->axis == 0)
        {
            MatrixFloatType m_field (mon_field + ((mon->position - x_start) * (mon->N1N2)), mon->N1, mon->N2);
            m_values += ((m_field.cast<std::complex<float>>()) * dtft_phase);
        }
        else
        {
            MatrixFloatStride m_field (
                mon_field + mon->yz_offset, 
                (x_stop - x_start),
                mon->N2,
                StrideType(mon->row_stride, mon->col_stride)
            );
            m_values.block(x_start, 0, x_stop - x_start, mon->N2) += ((m_field.cast<std::complex<float>>()) * dtft_phase);
        }
    }

    return 0;
}
