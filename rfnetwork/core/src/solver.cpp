
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

#define DATA_NDIM 3

#define MAX_MONITORS 20
#define MAX_PROBES 1000
#define MAX_THREADS 20

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
int n_monitors = 0;

Probe probes[MAX_PROBES];
int n_probes = 0;

std::thread threads[MAX_THREADS];
ThreadData thread_data[MAX_THREADS + 2];
int n_threads = 1;

// conditional variables used by the thread controller and update threads for synchronization
std::condition_variable cv;
std::condition_variable cv_th;

// shared variables protected by mutex
std::mutex mutex;
std::atomic<int> e_updates{0};
std::atomic<int> h_updates{0};
bool e_updates_done = false;
bool h_updates_done = false;
bool th_init_done = false;

static struct mbuffer_t m_pool = {NULL, NULL, 0};

long long get_milliseconds()
{

    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch()
                   ).count();

    return ms;
}

void mbuffer_init(float * base_addr, uint64_t size)
{
    m_pool.base_addr = base_addr;
    m_pool.next_addr = base_addr;
    m_pool.available_size = size;
}

/*
Reserves a contigious section of memory of given size (size is number of floats) 
and returns the pointer to the memory section.
Returns NULL if requested size does not fit within the memory block.
*/
float * mbuffer_allocate(uint64_t size)
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
            oss << "Invalid data array " << name << ". Expected shape on axis " << i << " of " << Nx;
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


int solver_init_fields(PyObject * py_mem, PyObject * coefficients, int Nx, int Ny, int Nz)
{

    // error check memory buffer
    std::ostringstream oss;
    PyArrayObject * mem_array = (PyArrayObject *) py_mem;

    if (PyArray_TYPE(mem_array) != NPY_FLOAT)
    {
        throw std::runtime_error("Invalid data array. Must be float type.");
        return NULL;
    }

    if (!(PyArray_FLAGS(mem_array) & NPY_ARRAY_C_CONTIGUOUS))
    {
        oss << "Invalid memory data array. Must be row ordered (C-style)";
        throw std::runtime_error(oss.str());
        return NULL;
    }

    if ((int) PyArray_NDIM(mem_array) != 1)
    {
        throw std::runtime_error("Invalid memory buffer array. Must be 1 dimension.");
        return NULL;
    }

    npy_intp * npy_shape = PyArray_SHAPE(mem_array); 

    // init memory buffer
    mbuffer_init((float *) PyArray_DATA(mem_array), (uint64_t) npy_shape[0]);

    // initialize ex pointers
    Ex.Nx = Nx;
    Ex.Ny = Ny + 1;
    Ex.Nz = Nz + 1;
    Ex.NyNz = Ex.Ny * Ex.Nz;

    // y and z endpoints do not get updated and don't have coefficients 
    Cx.Ny = Ny - 1; 
    Cx.Nz = Nz - 1;
    Cx.Cb_ex_y = get_solver_array(coefficients, "Cb_ex_y", Nx, Cx.Ny, Cx.Nz);
    Cx.Cb_ex_z = get_solver_array(coefficients, "Cb_ex_z", Nx, Cx.Ny, Cx.Nz);
    Cx.Ca_ex_y = get_solver_array(coefficients, "Ca_ex_y", Nx, Cx.Ny, Cx.Nz);
    Cx.Ca_ex_z = get_solver_array(coefficients, "Ca_ex_z", Nx, Cx.Ny, Cx.Nz);
    Cx.NyNz = Cx.Ny * Cx.Nz;

    // initialize ey pointers
    Ey.Nx = Nx;
    Ey.Ny = Ny;
    Ey.Nz = Nz + 1;
    Ey.NyNz = Ey.Ny * Ey.Nz;

    // x and z endpoints do not get updated and don't have coefficients 
    Cy.Ny = Ny; 
    Cy.Nz = Nz - 1;
    Cy.Cb_ey_z = get_solver_array(coefficients, "Cb_ey_z", Nx, Cy.Ny, Cy.Nz);
    Cy.Cb_ey_x = get_solver_array(coefficients, "Cb_ey_x", Nx, Cy.Ny, Cy.Nz);
    Cy.Ca_ey_z = get_solver_array(coefficients, "Ca_ey_z", Nx, Cy.Ny, Cy.Nz);
    Cy.Ca_ey_x = get_solver_array(coefficients, "Ca_ey_x", Nx, Cy.Ny, Cy.Nz);
    Cy.NyNz = Cy.Ny * Cy.Nz;
    // ensure coefficients at the end of the x-axis are zero to create a PEC boundary
    memset(Cy.Cb_ey_z + ((Nx - 1) * Cy.NyNz), 0, Cy.NyNz * sizeof(float));
    memset(Cy.Cb_ey_x + ((Nx - 1) * Cy.NyNz), 0, Cy.NyNz * sizeof(float));
    memset(Cy.Ca_ey_z + ((Nx - 1) * Cy.NyNz), 0, Cy.NyNz * sizeof(float));
    memset(Cy.Ca_ey_x + ((Nx - 1) * Cy.NyNz), 0, Cy.NyNz * sizeof(float));


    // initialize ez pointers
    Ez.Nx = Nx;
    Ez.Ny = Ny + 1;
    Ez.Nz = Nz;
    Ez.NyNz = Ez.Ny * Ez.Nz;

    // x and y endpoints do not get updated and don't have coefficients 
    Cz.Ny = Ny - 1; 
    Cz.Nz = Nz;
    Cz.Cb_ez_x = get_solver_array(coefficients, "Cb_ez_x", Nx, Cz.Ny, Cz.Nz);
    Cz.Cb_ez_y = get_solver_array(coefficients, "Cb_ez_y", Nx, Cz.Ny, Cz.Nz);
    Cz.Ca_ez_x = get_solver_array(coefficients, "Ca_ez_x", Nx, Cz.Ny, Cz.Nz);
    Cz.Ca_ez_y = get_solver_array(coefficients, "Ca_ez_y", Nx, Cz.Ny, Cz.Nz);
    Cz.NyNz = Cz.Ny * Cz.Nz;
    // ensure coefficients at the end of the x-axis are zero to create a PEC boundary
    memset(Cz.Cb_ez_x + ((Nx - 1) * Cz.NyNz), 0, Cz.NyNz * sizeof(float));
    memset(Cz.Cb_ez_y + ((Nx - 1) * Cz.NyNz), 0, Cz.NyNz * sizeof(float));
    memset(Cz.Ca_ez_x + ((Nx - 1) * Cz.NyNz), 0, Cz.NyNz * sizeof(float));
    memset(Cz.Ca_ez_y + ((Nx - 1) * Cz.NyNz), 0, Cz.NyNz * sizeof(float));

    // initialize hx pointers
    Hx.Nx = Nx;
    Hx.Ny = Ny;
    Hx.Nz = Nz;
    Hx.NyNz = Hx.Ny * Hx.Nz;

    Dx.Db_hx_y = get_solver_array(coefficients, "Db_hx_y", Nx, Hx.Ny, Hx.Nz);
    Dx.Db_hx_z = get_solver_array(coefficients, "Db_hx_z", Nx, Hx.Ny, Hx.Nz);
    Dx.Da_hx_y = get_solver_array(coefficients, "Da_hx_y", Nx, Hx.Ny, Hx.Nz);
    Dx.Da_hx_z = get_solver_array(coefficients, "Da_hx_z", Nx, Hx.Ny, Hx.Nz);
    // ensure coefficients at the end of the x-axis are zero to create a PEC boundary
    memset(Dx.Db_hx_y + ((Nx - 1) * Hx.Ny * Hx.Nz), 0, Hx.Ny * Hx.Nz * sizeof(float));
    memset(Dx.Db_hx_z + ((Nx - 1) * Hx.Ny * Hx.Nz), 0, Hx.Ny * Hx.Nz * sizeof(float));
    memset(Dx.Da_hx_y + ((Nx - 1) * Hx.Ny * Hx.Nz), 0, Hx.Ny * Hx.Nz * sizeof(float));
    memset(Dx.Da_hx_z + ((Nx - 1) * Hx.Ny * Hx.Nz), 0, Hx.Ny * Hx.Nz * sizeof(float));

    // initialize hy pointers
    Hy.Nx = Nx;
    Hy.Ny = Ny + 1;
    Hy.Nz = Nz;
    Hy.NyNz = Hy.Ny * Hy.Nz;

    Dy.Db_hy_z = get_solver_array(coefficients, "Db_hy_z", Nx, Hy.Ny, Hy.Nz);
    Dy.Db_hy_x = get_solver_array(coefficients, "Db_hy_x", Nx, Hy.Ny, Hy.Nz);
    Dy.Da_hy_z = get_solver_array(coefficients, "Da_hy_z", Nx, Hy.Ny, Hy.Nz);
    Dy.Da_hy_x = get_solver_array(coefficients, "Da_hy_x", Nx, Hy.Ny, Hy.Nz);

    // initialize hz pointers
    Hz.Nx = Nx;
    Hz.Ny = Ny;
    Hz.Nz = Nz + 1;
    Hz.NyNz = Hz.Ny * Hz.Nz;

    Dz.Db_hz_x = get_solver_array(coefficients, "Db_hz_x", Nx, Hz.Ny, Hz.Nz);
    Dz.Db_hz_y = get_solver_array(coefficients, "Db_hz_y", Nx, Hz.Ny, Hz.Nz);
    Dz.Da_hz_x = get_solver_array(coefficients, "Da_hz_x", Nx, Hz.Ny, Hz.Nz);
    Dz.Da_hz_y = get_solver_array(coefficients, "Da_hz_y", Nx, Hz.Ny, Hz.Nz);

    return 0;
}

int solver_init_monitors(PyObject * py_monitors, int Nt)
{
    // initialize field monitors
    n_monitors = (int) PyList_Size(py_monitors);

    if (n_monitors > MAX_MONITORS)
    {
        throw std::runtime_error("Exceeded maximum number of monitors.");
    }

    // shapes of field components
    int Nx[6] = {Ex.Nx, Ey.Nx+1, Ez.Nx+1, Hx.Nx+1, Hy.Nx, Hz.Nx};
    int Ny[6] = {Ex.Ny, Ey.Ny, Ez.Ny, Hx.Ny, Hy.Ny, Hz.Ny};
    int Nz[6] = {Ex.Nz, Ey.Nz, Ez.Nz, Hx.Nz, Hy.Nz, Hz.Nz};

    PyObject* py_mon;

    int n1;
    int n2; 

    for (int m = 0; m < n_monitors; m++)
    {   
        py_mon = PyList_GetItem(py_monitors, m);

        // number of elements in time dimension
        int n_step = PyLong_AsLong(PyDict_GetItemString(py_mon, "n_step"));
        int Nm = (Nt / n_step) + 1;
        
        int axis = PyLong_AsLong(PyDict_GetItemString(py_mon, "axis"));
        int field = PyLong_AsLong(PyDict_GetItemString(py_mon, "field"));
        
        if (axis == 0)
        {
            n1 = Ny[field];
            n2 = Nz[field];
        }
        else if (axis == 1)
        {
            n1 = Nx[field];
            n2 = Nz[field];
        }
        else
        {
            n1 = Nx[field];
            n2 = Ny[field];
        }

        monitors[m].values = get_solver_array(py_mon, "values", Nm, n1, n2);
        monitors[m].field_type = field;
        monitors[m].position = PyLong_AsLong(PyDict_GetItemString(py_mon, "position"));
        monitors[m].n_step = n_step;
        monitors[m].axis = axis;
        monitors[m].N1 = n1;
        monitors[m].N2 = n2;
        monitors[m].Nx = Nx[field];
        monitors[m].Ny = Ny[field];
        monitors[m].Nz = Nz[field];
        monitors[m].NyNz = Ny[field] * Nz[field];
        monitors[m].N1N2 = n1 * n2;

    }

    return 0;
}

int solver_init_probes(PyObject * py_probes, int Nt)
{
    // shapes of field components
    // int Nx[6] = {Ex.Nx, Ey.Nx, Ez.Nx, Hx.Nx, Hy.Nx, Hz.Nx};
    int Ny[6] = {Ex.Ny, Ey.Ny, Ez.Ny, Hx.Ny, Hy.Ny, Hz.Ny};
    int Nz[6] = {Ex.Nz, Ey.Nz, Ez.Nz, Hx.Nz, Hy.Nz, Hz.Nz};

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

        int field_type = PyLong_AsLong(PyDict_GetItemString(probe_dict, "field"));
        probes[s].field_type = field_type;

        // if component's first index is at x=0 instead of x=0.5, the first usable component
        // is at index=1
        if ((field_type == 1) || (field_type == 2) || (field_type == 3))
        {
            probes[s].x_cell -= 1;
        }

        if (probes[s].x_cell < 0)
        {
            throw std::runtime_error("Source cannot be placed on grid edge");
        }

        probes[s].yz_offset = (
            ((int) PyLong_AsLong(PyList_GetItem(py_idx, 1)) * Nz[field_type]) +
            ((int) PyLong_AsLong(PyList_GetItem(py_idx, 2)))
        );

        // size of yz grid
        probes[s].NyNz = (Ny[field_type] * Nz[field_type]);
        
        // flag probe if it provides source values
        probes[s].is_source = PyDict_GetItemString(probe_dict, "is_source");
    }

    return 0;
}

int solver_run(int Nt, int n_th)
{
    n_threads = n_th;
    // error check number of threads
    if (n_threads < 0 || n_threads > MAX_THREADS)
    {
        throw std::runtime_error("Invalid number of threads.");
    }

    // number of x slices computed by each thread
    int n_batch = Ex.Nx / n_threads;
    // remainder of batch size
    int r_batch = Ex.Nx % n_threads;

    int x_start_th = 0;
    int x_stop_th = 0;

    // start controller thread
    std::thread control_th = std::thread(solver_controller, Nt, n_threads);

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
        threads[t] = std::thread(solver_thread, x_start_th, x_stop_th, Nt, t+1);
    }

    // allocate dummy data for endpoints of thread grids. 
    thread_data[0].ez = mbuffer_allocate(Ez.Ny * Ez.Nz);
    thread_data[0].ey = mbuffer_allocate(Ey.Ny * Ey.Nz);
    thread_data[n_threads+1].hy = mbuffer_allocate(Hy.Ny * Hy.Nz);
    thread_data[n_threads+1].hz = mbuffer_allocate(Hz.Ny * Hz.Nz);

    // wait for all threads to complete
    for (int t = 0; t < n_threads; t++)
    {
        threads[t].join();
    }

    control_th.join();

    return 0;
}


// thread responsible for synchronizing field update threads
void solver_controller(int Nt, int n_threads)
{
    // std::stringstream msg;
    // msg.str("");
    // msg.clear();
    // msg << "Controller Start... \n";
    // std::cout << msg.str();

    // wait until all threads have signaled they are done with memory allocation
    {
        // get lock on mutex while shared variables are modified
        std::unique_lock<std::mutex> lock(mutex);
        e_updates = 0;

        // wait until all threads are done with memory allocation
        cv.wait(lock, [n_threads] { return e_updates.load() == n_threads; });
        e_updates = 0;
        th_init_done = true;
        cv_th.notify_all();
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
            cv.wait(lock, [n_threads] { return e_updates.load() == n_threads; });
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
            cv.wait(lock, [n_threads] { return h_updates.load() == n_threads; });
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

}

void solver_thread(int x_start, int x_stop, int Nt, int thread_idx)
{
    // each grid cell contains the ex, ey, hz components in the middle of the x axis of the cell, and the
    // ey, ez, hx components on the RIGHT side of the cell. The first cell does not update the ey, ez, hx compoennts
    // on the left

    // std::stringstream msg;
    // msg << "Starting Thread " << thread_idx << " " << x_start << " " << x_stop << "\n";
    // std::cout << msg.str();

    // long long k = 0;
    int x_offset;
    // number of updated field components
    int Nx = x_stop - x_start;

    int Ny;
    int Nz;

    // allocate memory for this thread's grid.
    // only Nx components are created for all fields, the Ey, Ez, and Hx components that have one extra 
    // component do not track the fields on the first index. They are not updated as they are on the edge of the grid
    // and always remain at zero.
    float * p_ex_y = mbuffer_allocate(Nx * Ex.Ny * Ex.Nz); // 
    float * p_ex_z = mbuffer_allocate(Nx * Ex.Ny * Ex.Nz); // 
    float * p_ex   = mbuffer_allocate(Nx * Ex.Ny * Ex.Nz); // 

    float * p_ey_z = mbuffer_allocate(Nx * Ey.Ny * Ey.Nz); //  
    float * p_ey_x = mbuffer_allocate(Nx * Ey.Ny * Ey.Nz); //  
    float * p_ey   = mbuffer_allocate(Nx * Ey.Ny * Ey.Nz); // 

    float * p_ez_x = mbuffer_allocate(Nx * Ez.Ny * Ez.Nz); //  
    float * p_ez_y = mbuffer_allocate(Nx * Ez.Ny * Ez.Nz); //  
    float * p_ez   = mbuffer_allocate(Nx * Ez.Ny * Ez.Nz); // 

    float * p_hx_y = mbuffer_allocate(Nx * Hx.Ny * Hx.Nz); //  
    float * p_hx_z = mbuffer_allocate(Nx * Hx.Ny * Hx.Nz); //  
    float * p_hx   = mbuffer_allocate(Nx * Hx.Ny * Hx.Nz); //  

    float * p_hy_z = mbuffer_allocate(Nx * Hy.Ny * Hy.Nz); //  
    float * p_hy_x = mbuffer_allocate(Nx * Hy.Ny * Hy.Nz); //  
    float * p_hy   = mbuffer_allocate(Nx * Hy.Ny * Hy.Nz); //  

    float * p_hz_x = mbuffer_allocate(Nx * Hz.Ny * Hz.Nz); //  
    float * p_hz_y = mbuffer_allocate(Nx * Hz.Ny * Hz.Nz); //  
    float * p_hz   = mbuffer_allocate(Nx * Hz.Ny * Hz.Nz); //  

    // populate thread data
    thread_data[thread_idx].hy = p_hy;
    thread_data[thread_idx].hz = p_hz;
    thread_data[thread_idx].ey = p_ey + ((Nx - 1) * Ey.NyNz);
    thread_data[thread_idx].ez = p_ez + ((Nx - 1) * Ez.NyNz);

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

    for (int i = 0; i < n_probes; i++)
    {   
        p = &(probes[i]);
        // if probe is inside the grid for this thread
        if (((p->x_cell) >= x_start) && ((p->x_cell) < x_stop))
        {
            // add e-field probe
            if ((p->field_type) < 3)
            {
                e_probes.push_back(p);
            }
            // add h-field probe
            else 
            {
                h_probes.push_back(p);
            }
            // compute the pointer for the probe in the grid. x_index needs to account for the starting position of 
            // the grid.
            x_offset = (((p->x_cell) - x_start) * (p->NyNz)) + (p->yz_offset);
            p->field_p = (fields_base[p->field_type]) + x_offset;
            p->field_s1_p = (fields_sp1_base[p->field_type]) + x_offset;
            p->field_s2_p = (fields_sp2_base[p->field_type]) + x_offset;

        }
    }

    {
        // lock the mutex while updating shared variable, also ensures that only one thread sends a notification
        // to the controller at a time, preventing missed notifications.
        std::unique_lock<std::mutex> lock(mutex);
        ++e_updates;

        cv.notify_all(); 
        // wait for all threads to reach this point before moving on to h-field updates.
        cv_th.wait(lock, [] { return th_init_done; });
    }

    // main time stepping loop
    for (int n = 0; n < Nt; n++)
    {

        // operate on a single slice of the field on the x axis
        for (int x = 0; x < Nx; x++)
        {   
            x_offset = x * Ex.NyNz;
            MatrixFloatType ex_y (p_ex_y + x_offset, Ex.Ny, Ex.Nz);
            MatrixFloatType ex_z (p_ex_z + x_offset, Ex.Ny, Ex.Nz);
            MatrixFloatType ex   (p_ex   + x_offset, Ex.Ny, Ex.Nz);
            
            x_offset = x * Ey.NyNz;
            MatrixFloatType ey_z (p_ey_z + x_offset, Ey.Ny, Ey.Nz);
            MatrixFloatType ey_x (p_ey_x + x_offset, Ey.Ny, Ey.Nz);
            MatrixFloatType ey   (p_ey   + x_offset, Ey.Ny, Ey.Nz);

            x_offset = x * Ez.NyNz;
            MatrixFloatType ez_x (p_ez_x + x_offset, Ez.Ny, Ez.Nz);
            MatrixFloatType ez_y (p_ez_y + x_offset, Ez.Ny, Ez.Nz);
            MatrixFloatType ez   (p_ez   + x_offset, Ez.Ny, Ez.Nz);

            // h-fields
            x_offset = x * Hx.NyNz;
            MatrixFloatType hx   (p_hx   + x_offset, Hx.Ny, Hx.Nz);

            x_offset = x * Hy.NyNz;
            MatrixFloatType hy   (p_hy   + x_offset, Hy.Ny, Hy.Nz);

            x_offset = x * Hz.NyNz;
            MatrixFloatType hz   (p_hz   + x_offset, Hz.Ny, Hz.Nz);

            // ex coefficients
            x_offset = (x_start + x) * Cx.NyNz;
            MatrixFloatType Cb_ex_y (Cx.Cb_ex_y + x_offset, Cx.Ny, Cx.Nz);
            MatrixFloatType Cb_ex_z (Cx.Cb_ex_z + x_offset, Cx.Ny, Cx.Nz);

            MatrixFloatType Ca_ex_y (Cx.Ca_ex_y + x_offset, Cx.Ny, Cx.Nz);
            MatrixFloatType Ca_ex_z (Cx.Ca_ex_z + x_offset, Cx.Ny, Cx.Nz);

            // ey coefficients. 
            x_offset = (x_start + x) * Cy.NyNz;
            MatrixFloatType Cb_ey_z (Cy.Cb_ey_z + x_offset, Cy.Ny, Cy.Nz);
            MatrixFloatType Cb_ey_x (Cy.Cb_ey_x + x_offset, Cy.Ny, Cy.Nz);

            MatrixFloatType Ca_ey_z (Cy.Ca_ey_z + x_offset, Cy.Ny, Cy.Nz);
            MatrixFloatType Ca_ey_x (Cy.Ca_ey_x + x_offset, Cy.Ny, Cy.Nz);
            
            // ez coefficients
            x_offset = (x_start + x) * Cz.NyNz;
            MatrixFloatType Cb_ez_x (Cz.Cb_ez_x + x_offset, Cz.Ny, Cz.Nz);
            MatrixFloatType Cb_ez_y (Cz.Cb_ez_y + x_offset, Cz.Ny, Cz.Nz);

            MatrixFloatType Ca_ez_x (Cz.Ca_ez_x + x_offset, Cz.Ny, Cz.Nz);
            MatrixFloatType Ca_ez_y (Cz.Ca_ez_y + x_offset, Cz.Ny, Cz.Nz);


            // next cell components. Dummy cells provide all zero components for the threads at the end points
            // of the grid
            p_hz_1 = (x < (Nx - 1)) ? p_hz + ((x + 1) * Hz.NyNz) : (thread_data[thread_idx + 1]).hz;
            p_hy_1 = (x < (Nx - 1)) ? p_hy + ((x + 1) * Hy.NyNz) : (thread_data[thread_idx + 1]).hy;
            MatrixFloatType hz_1 (p_hz_1, Hz.Ny, Hz.Nz);
            MatrixFloatType hy_1 (p_hy_1, Hy.Ny, Hy.Nz);

            // ----------------- update ex -------------------------- //
            Ny = Cx.Ny;
            Nz = Cx.Nz;
            // ex_y update
            // ex_yd = Cb_ex_y * np.diff(hz, axis=1)[:, :, 1:-1]
            // ex_y[:, 1:-1, 1:-1] = (Ca_ex_y * ex_y[:, 1:-1, 1:-1]) + ex_yd
            ex_y.block(1, 1, Ny, Nz) = Ca_ex_y.cwiseProduct(ex_y.block(1, 1, Ny, Nz)) + (
                Cb_ex_y.cwiseProduct((hz.bottomRows(Ny) - hz.topRows(Ny)).block(0, 1, Ny, Nz))
            );

            // ex_z update
            // ex_zd = Cb_ex_z * np.diff(hy, axis=2)[:, 1:-1, :]
            // ex_z[:, 1:-1, 1:-1] = (Ca_ex_z * ex_z[:, 1:-1, 1:-1]) + ex_zd
            ex_z.block(1, 1, Ny, Nz) = Ca_ex_z.cwiseProduct(ex_z.block(1, 1, Ny, Nz)) + (
                Cb_ex_z.cwiseProduct((hy.rightCols(Nz) - hy.leftCols(Nz)).block(1, 0, Ny, Nz))
            );

            // ----------------- update ey -------------------------- //
            Ny = Cy.Ny;
            Nz = Cy.Nz;
            // ey_z update
            // ey_zd = Cb_ey_z * np.diff(hx, axis=2)[1:-1, :, :]
            // ey_z[1:-1, :, 1:-1] = (Ca_ey_z * ey_z[1:-1, :, 1:-1]) + ey_zd
            ey_z.block(0, 1, Ny, Nz) = Ca_ey_z.cwiseProduct(ey_z.block(0, 1, Ny, Nz)) + (
                Cb_ey_z.cwiseProduct(hx.rightCols(Nz) - hx.leftCols(Nz))
            );

            // ey_x update
            // ey_xd = Cb_ey_x * np.diff(hz, axis=0)[:, :, 1:-1]
            // ey_x[1:-1, :, 1:-1] = (Ca_ey_x * ey_x[1:-1, :, 1:-1]) + ey_xd
            ey_x.block(0, 1, Ny, Nz) = Ca_ey_x.cwiseProduct(ey_x.block(0, 1, Ny, Nz)) + (
                Cb_ey_x.cwiseProduct((hz_1 - hz).block(0, 1, Ny, Nz))
            );
            
            // ----------------- update ez -------------------------- //
            Ny = Cz.Ny;
            Nz = Cz.Nz;
            // ex_z update
            // ez_xd = Cb_ez_x * np.diff(hy, axis=0)[:, 1:-1, :]
            // ez_x[1:-1, 1:-1, :] = (Ca_ez_x * ez_x[1:-1, 1:-1, :]) + ez_xd
            // get hy components on either side of x-slice, the hz component below ez is in the same cell,
            ez_x.block(1, 0, Ny, Nz) = Ca_ez_x.cwiseProduct(ez_x.block(1, 0, Ny, Nz)) + (
                Cb_ez_x.cwiseProduct((hy_1 - hy).block(1, 0, Ny, Nz))
            );

            // update ez_y
            // ez_yd = Cb_ez_y * np.diff(hx, axis=1)[1:-1, :, :]
            // ez_y[1:-1, 1:-1, :] = (Ca_ez_y * ez_y[1:-1, 1:-1, :]) + ez_yd
            ez_y.block(1, 0, Ny, Nz) = Ca_ez_y.cwiseProduct(ez_y.block(1, 0, Ny, Nz)) + (
                Cb_ez_y.cwiseProduct(hx.bottomRows(Ny) - hx.topRows(Ny))
            );


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
            cv_th.wait(lock, [] { return e_updates_done; });
        }

        // h-updates
        for (int x = 0; x < Nx; x++)
        {   
            x_offset = x * Hx.NyNz;
            MatrixFloatType hx_y (p_hx_y + x_offset, Hx.Ny, Hx.Nz);
            MatrixFloatType hx_z (p_hx_z + x_offset, Hx.Ny, Hx.Nz);
            MatrixFloatType hx   (p_hx   + x_offset, Hx.Ny, Hx.Nz);

            x_offset = x * Hy.NyNz;
            MatrixFloatType hy_z (p_hy_z + x_offset, Hy.Ny, Hy.Nz);
            MatrixFloatType hy_x (p_hy_x + x_offset, Hy.Ny, Hy.Nz);
            MatrixFloatType hy   (p_hy   + x_offset, Hy.Ny, Hy.Nz);

            x_offset = x * Hz.NyNz;
            MatrixFloatType hz_x (p_hz_x + x_offset, Hz.Ny, Hz.Nz);
            MatrixFloatType hz_y (p_hz_y + x_offset, Hz.Ny, Hz.Nz);
            MatrixFloatType hz   (p_hz   + x_offset, Hz.Ny, Hz.Nz);

            // e-fields 
            x_offset = x * Ex.NyNz;
            MatrixFloatType ex   (p_ex   + x_offset, Ex.Ny, Ex.Nz);

            x_offset = x * Ey.NyNz;
            MatrixFloatType ey   (p_ey   + x_offset, Ey.Ny, Ey.Nz);

            x_offset = x * Ez.NyNz;
            MatrixFloatType ez   (p_ez   + x_offset, Ez.Ny, Ez.Nz);

            // hx coefficients
            x_offset = (x_start + x) * Hx.NyNz;
            MatrixFloatType Db_hx_y (Dx.Db_hx_y + x_offset, Hx.Ny, Hx.Nz);
            MatrixFloatType Db_hx_z (Dx.Db_hx_z + x_offset, Hx.Ny, Hx.Nz);

            MatrixFloatType Da_hx_y (Dx.Da_hx_y + x_offset, Hx.Ny, Hx.Nz);
            MatrixFloatType Da_hx_z (Dx.Da_hx_z + x_offset, Hx.Ny, Hx.Nz);

            // hy coefficients
            x_offset = (x_start + x) * Hy.NyNz;
            MatrixFloatType Db_hy_z (Dy.Db_hy_z + x_offset, Hy.Ny, Hy.Nz);
            MatrixFloatType Db_hy_x (Dy.Db_hy_x + x_offset, Hy.Ny, Hy.Nz);

            MatrixFloatType Da_hy_z (Dy.Da_hy_z + x_offset, Hy.Ny, Hy.Nz);
            MatrixFloatType Da_hy_x (Dy.Da_hy_x + x_offset, Hy.Ny, Hy.Nz);

            // hz coefficients
            x_offset = (x_start + x) * Hz.NyNz;
            MatrixFloatType Db_hz_x (Dz.Db_hz_x + x_offset, Hz.Ny, Hz.Nz);
            MatrixFloatType Db_hz_y (Dz.Db_hz_y + x_offset, Hz.Ny, Hz.Nz);

            MatrixFloatType Da_hz_x (Dz.Da_hz_x + x_offset, Hz.Ny, Hz.Nz);
            MatrixFloatType Da_hz_y (Dz.Da_hz_y + x_offset, Hz.Ny, Hz.Nz);

            // previous cell components. Dummy cells provide all zero components for the threads at the end points
            // of the grid
            p_ey_0 = (x > 0) ? p_ey + ((x - 1) * Ey.NyNz) : (thread_data[thread_idx - 1]).ey;
            p_ez_0 = (x > 0) ? p_ez + ((x - 1) * Ez.NyNz) : (thread_data[thread_idx - 1]).ez;
            MatrixFloatType ey_0 (p_ey_0, Ey.Ny, Ey.Nz);
            MatrixFloatType ez_0 (p_ez_0, Ez.Ny, Ez.Nz);

            // ----------------- update hx -------------------------- //
            Ny = Hx.Ny;
            Nz = Hx.Nz;
            // hx_y update
            // hx_yd = Db_hx_y * np.diff(ez, axis=1)
            // hx_y = Da_hx_y * hx_y + hx_yd
            hx_y = Da_hx_y.cwiseProduct(hx_y) + (
                Db_hx_y.cwiseProduct(ez.bottomRows(Ny) - ez.topRows(Ny))
            );

            // hx_z update
            // hx_zd = Db_hx_z * np.diff(ey, axis=2)
            // hx_z = Da_hx_z * hx_z + hx_zd
            hx_z = Da_hx_z.cwiseProduct(hx_z) + (
                Db_hx_z.cwiseProduct(ey.rightCols(Nz) - ey.leftCols(Nz))
            );
            
            // ----------------- update hy -------------------------- //
            Ny = Hy.Ny;
            Nz = Hy.Nz;
            // hy_z update
            // hy_zd = Db_hy_z * np.diff(ex, axis=2)
            // hy_z = Da_hy_z * hy_z + hy_zd
            hy_z = Da_hy_z.cwiseProduct(hy_z) + (
                Db_hy_z.cwiseProduct(ex.rightCols(Nz) - ex.leftCols(Nz))
            );

            // update hy_x
            // hy_xd = Db_hy_x * np.diff(ez, axis=0)
            // hy_x = Da_hy_x * hy_x + hy_xd
            hy_x = Da_hy_x.cwiseProduct(hy_x) + (
                Db_hy_x.cwiseProduct(ez - ez_0)
            );

            // ----------------- update hz -------------------------- //
            Ny = Hz.Ny;
            Nz = Hz.Nz;
            // hz_x update
            // hz_xd = Db_hz_x * np.diff(ey, axis=0) 
            // hz_x = Da_hz_x * hz_x + hz_xd
            hz_x = Da_hz_x.cwiseProduct(hz_x) + (
                Db_hz_x.cwiseProduct(ey - ey_0)
            );

            // update hz_y
            // hz_yd = Db_hz_y * np.diff(ex, axis=1)
            // hz_y = Da_hz_y * hz_y + hz_yd
            hz_y = Da_hz_y.cwiseProduct(hz_y) + (
                Db_hz_y.cwiseProduct(ex.bottomRows(Ny) - ex.topRows(Ny))
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
            cv_th.wait(lock, [] { return h_updates_done; });
        }

        // update monitors
        for (int m = 0; m < n_monitors; m++)
        {
            Monitor mon = monitors[m];
            // pointer to field in grid
            mon_field = fields_base[mon.field_type];

            if ((n > 0) && ((n % (mon.n_step)) != 0))
            {
                continue;
            }

            int m_n = n / mon.n_step;

            MatrixFloatType m_values (mon.values + (m_n * (mon.N1N2)), mon.N1, mon.N2);
            
            if (mon.axis == 0)
            {
                // only update x slice monitor if it is within bounds of the thread.
                // position is subtracted by one because the monitor contains all field values along x, 
                // and the thread grid skips the first component along x, the first cell is at x=1 in the global grid
                if (((mon.position -1) < x_start) || ((mon.position -1) >= x_stop))
                {
                    continue;
                }
                MatrixFloatType m_field (mon_field + (mon.position * (mon.NyNz)), mon.Ny, mon.Nz);
                m_values = m_field;
            }

            else if (mon.axis == 1)
            {
                // components on the edge of the grid that have +1 components from the grid will not be captured here,
                // but they are not updated and will be zero. 
                // as long as the monitor values were initialized in python to zero, this is a non-issue.
                for (int i = x_start; i < x_stop; i++)
                {
                    MatrixFloatType m_field (mon_field + ((i - x_start) * (mon.NyNz)), mon.Ny, mon.Nz);
                    // monitor grid contains the components at the x=0 edge, add one to the x position.
                    m_values.row(i+1) = m_field.row(mon.position);
                }
            }

            else // (m.axis == 2)
            {
                for (int i = x_start; i < x_stop; i++)
                {
                    MatrixFloatType m_field (mon_field + ((i - x_start) * (mon.NyNz)), mon.Ny, mon.Nz);
                    m_values.row(i+1) = m_field.col(mon.position);
                }
            }
            
        }

    }

}
