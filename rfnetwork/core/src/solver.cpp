
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


#define DATA_NDIM 3
#define MAX_MONITORS 20
#define MAX_SOURCES 50
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

Source sources[MAX_SOURCES];
int n_sources = 0;

std::thread threads[MAX_THREADS];

// conditional variables used by the thread controller and update threads for synchronization
std::condition_variable cv;
std::condition_variable cv_th;

// shared variables protected by mutex
std::mutex mutex;
std::atomic<int> e_updates{0};
std::atomic<int> h_updates{0};
bool e_updates_done = false;
bool h_updates_done = false;

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


int solver_init_fields(PyObject * fields, PyObject * coefficients, int Nx, int Ny, int Nz)
{
    // initialize ex pointers
    Ex.Nx = Nx;
    Ex.Ny = Ny + 1;
    Ex.Nz = Nz + 1;
    Ex.ex_y = get_solver_array(fields, "ex_y", Ex.Nx, Ex.Ny, Ex.Nz);
    Ex.ex_z = get_solver_array(fields, "ex_z", Ex.Nx, Ex.Ny, Ex.Nz);
    Ex.ex   = get_solver_array(fields, "ex",   Ex.Nx, Ex.Ny, Ex.Nz);
    Ex.NyNz = Ex.Ny * Ex.Nz;

    // y and z endpoints do not get updated and don't have coefficients 
    Cx.Nx = Nx; 
    Cx.Ny = Ny - 1; 
    Cx.Nz = Nz - 1;
    Cx.Cb_ex_y = get_solver_array(coefficients, "Cb_ex_y", Cx.Nx, Cx.Ny, Cx.Nz);
    Cx.Cb_ex_z = get_solver_array(coefficients, "Cb_ex_z", Cx.Nx, Cx.Ny, Cx.Nz);
    Cx.Ca_ex_y = get_solver_array(coefficients, "Ca_ex_y", Cx.Nx, Cx.Ny, Cx.Nz);
    Cx.Ca_ex_z = get_solver_array(coefficients, "Ca_ex_z", Cx.Nx, Cx.Ny, Cx.Nz);
    Cx.NyNz = Cx.Ny * Cx.Nz;

    // initialize ey pointers
    Ey.Nx = Nx + 1;
    Ey.Ny = Ny;
    Ey.Nz = Nz + 1;
    Ey.ey_z = get_solver_array(fields, "ey_z", Ey.Nx, Ey.Ny, Ey.Nz);
    Ey.ey_x = get_solver_array(fields, "ey_x", Ey.Nx, Ey.Ny, Ey.Nz);
    Ey.ey   = get_solver_array(fields, "ey",   Ey.Nx, Ey.Ny, Ey.Nz);
    Ey.NyNz = Ey.Ny * Ey.Nz;

    // x and z endpoints do not get updated and don't have coefficients 
    Cy.Nx = Nx - 1; 
    Cy.Ny = Ny; 
    Cy.Nz = Nz - 1;
    Cy.Cb_ey_z = get_solver_array(coefficients, "Cb_ey_z", Cy.Nx, Cy.Ny, Cy.Nz);
    Cy.Cb_ey_x = get_solver_array(coefficients, "Cb_ey_x", Cy.Nx, Cy.Ny, Cy.Nz);
    Cy.Ca_ey_z = get_solver_array(coefficients, "Ca_ey_z", Cy.Nx, Cy.Ny, Cy.Nz);
    Cy.Ca_ey_x = get_solver_array(coefficients, "Ca_ey_x", Cy.Nx, Cy.Ny, Cy.Nz);
    Cy.NyNz = Cy.Ny * Cy.Nz;

    // initialize ez pointers
    Ez.Nx = Nx + 1;
    Ez.Ny = Ny + 1;
    Ez.Nz = Nz;
    Ez.ez_x = get_solver_array(fields, "ez_x", Ez.Nx, Ez.Ny, Ez.Nz);
    Ez.ez_y = get_solver_array(fields, "ez_y", Ez.Nx, Ez.Ny, Ez.Nz);
    Ez.ez   = get_solver_array(fields, "ez",   Ez.Nx, Ez.Ny, Ez.Nz);
    Ez.NyNz = Ez.Ny * Ez.Nz;

    // x and y endpoints do not get updated and don't have coefficients 
    Cz.Nx = Nx - 1; 
    Cz.Ny = Ny - 1; 
    Cz.Nz = Nz;
    Cz.Cb_ez_x = get_solver_array(coefficients, "Cb_ez_x", Cz.Nx, Cz.Ny, Cz.Nz);
    Cz.Cb_ez_y = get_solver_array(coefficients, "Cb_ez_y", Cz.Nx, Cz.Ny, Cz.Nz);
    Cz.Ca_ez_x = get_solver_array(coefficients, "Ca_ez_x", Cz.Nx, Cz.Ny, Cz.Nz);
    Cz.Ca_ez_y = get_solver_array(coefficients, "Ca_ez_y", Cz.Nx, Cz.Ny, Cz.Nz);
    Cz.NyNz = Cz.Ny * Cz.Nz;

    // initialize hx pointers
    Hx.Nx = Nx + 1;
    Hx.Ny = Ny;
    Hx.Nz = Nz;
    Hx.hx_y = get_solver_array(fields, "hx_y", Hx.Nx, Hx.Ny, Hx.Nz);
    Hx.hx_z = get_solver_array(fields, "hx_z", Hx.Nx, Hx.Ny, Hx.Nz);
    Hx.hx   = get_solver_array(fields, "hx",   Hx.Nx, Hx.Ny, Hx.Nz);
    Hx.NyNz = Hx.Ny * Hx.Nz;

    Dx.Db_hx_y = get_solver_array(coefficients, "Db_hx_y", Hx.Nx, Hx.Ny, Hx.Nz);
    Dx.Db_hx_z = get_solver_array(coefficients, "Db_hx_z", Hx.Nx, Hx.Ny, Hx.Nz);
    Dx.Da_hx_y = get_solver_array(coefficients, "Da_hx_y", Hx.Nx, Hx.Ny, Hx.Nz);
    Dx.Da_hx_z = get_solver_array(coefficients, "Da_hx_z", Hx.Nx, Hx.Ny, Hx.Nz);

    // initialize hy pointers
    Hy.Nx = Nx;
    Hy.Ny = Ny + 1;
    Hy.Nz = Nz;
    Hy.hy_z = get_solver_array(fields, "hy_z", Hy.Nx, Hy.Ny, Hy.Nz);
    Hy.hy_x = get_solver_array(fields, "hy_x", Hy.Nx, Hy.Ny, Hy.Nz);
    Hy.hy   = get_solver_array(fields, "hy",   Hy.Nx, Hy.Ny, Hy.Nz);
    Hy.NyNz = Hy.Ny * Hy.Nz;

    Dy.Db_hy_z = get_solver_array(coefficients, "Db_hy_z", Hy.Nx, Hy.Ny, Hy.Nz);
    Dy.Db_hy_x = get_solver_array(coefficients, "Db_hy_x", Hy.Nx, Hy.Ny, Hy.Nz);
    Dy.Da_hy_z = get_solver_array(coefficients, "Da_hy_z", Hy.Nx, Hy.Ny, Hy.Nz);
    Dy.Da_hy_x = get_solver_array(coefficients, "Da_hy_x", Hy.Nx, Hy.Ny, Hy.Nz);

    // initialize hz pointers
    Hz.Nx = Nx;
    Hz.Ny = Ny;
    Hz.Nz = Nz + 1;
    Hz.hz_x = get_solver_array(fields, "hz_x", Hz.Nx, Hz.Ny, Hz.Nz);
    Hz.hz_y = get_solver_array(fields, "hz_y", Hz.Nx, Hz.Ny, Hz.Nz);
    Hz.hz   = get_solver_array(fields, "hz",   Hz.Nx, Hz.Ny, Hz.Nz);
    Hz.NyNz = Hz.Ny * Hz.Nz;

    Dz.Db_hz_x = get_solver_array(coefficients, "Db_hz_x", Hz.Nx, Hz.Ny, Hz.Nz);
    Dz.Db_hz_y = get_solver_array(coefficients, "Db_hz_y", Hz.Nx, Hz.Ny, Hz.Nz);
    Dz.Da_hz_x = get_solver_array(coefficients, "Da_hz_x", Hz.Nx, Hz.Ny, Hz.Nz);
    Dz.Da_hz_y = get_solver_array(coefficients, "Da_hz_y", Hz.Nx, Hz.Ny, Hz.Nz);

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

    // pointer to all field components
    float * field_p[6] = {Ex.ex, Ey.ey, Ez.ez, Hx.hx, Hy.hy, Hz.hz};

    // shapes of field components
    int Nx[6] = {Ex.Nx, Ey.Nx, Ez.Nx, Hx.Nx, Hy.Nx, Hz.Nx};
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
        monitors[m].field = field_p[field];
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

int solver_init_sources(PyObject * py_sources, int Nt)
{
    n_sources = (int) PyList_Size(py_sources);

    if (n_sources > MAX_SOURCES)
    {
        throw std::runtime_error("Exceeded maximum number of sources.");
    }

    // get waveform data for each component where a source is present
    for (int s = 0; s < n_sources; s++) 
    {

        // verify each item in the list is a python dictionary
        PyObject * source_dict = PyList_GetItem(py_sources, s);
        if (!PyDict_Check(source_dict)) {
            throw std::runtime_error("Expected a list of source dictionaries");
        }

        // get waveform data
        sources[s].values = get_source_array(source_dict, Nt);

        // get component indices for ez
        PyObject* py_idx = PyDict_GetItemString(source_dict, "idx");

        int offset = (
            ((int) PyLong_AsLong(PyList_GetItem(py_idx, 0)) * Ez.NyNz) + 
            ((int) PyLong_AsLong(PyList_GetItem(py_idx, 1)) * Ez.Nz) +
            ((int) PyLong_AsLong(PyList_GetItem(py_idx, 2)))
        );
        
        sources[s].ez = Ez.ez + offset;
        sources[s].ez_x = Ez.ez_x + offset;
        sources[s].ez_y = Ez.ez_y + offset;

        sources[s].x_idx = (int) PyLong_AsLong(PyList_GetItem(py_idx, 0));
    }

    return 0;
}

int solver_run(int Nt, int n_threads)
{

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

        threads[t] = std::thread(solver_thread, x_start_th, x_stop_th, Nt, t);
    }

    // start controller thread
    std::thread control_th = std::thread(solver_controller, Nt, n_threads);

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

        // reset e update counter and signal to threads that they can move on to h updates
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
    // std::stringstream msg;
    // msg << "Starting Thread " << thread_idx << " " << x_start << " " << x_stop << "\n";
    // std::cout << msg.str();

    for (int n = 0; n < Nt; n++)
    {
        solver_update_ex(x_start, x_stop);
        solver_update_ey(x_start, x_stop);
        solver_update_ez(x_start, x_stop);

        // update sources
        for (int s = 0; s < n_sources; s++)
        {   
            if ((sources[s].x_idx < x_start) || (sources[s].x_idx >= x_stop))
            {
                continue;
            }
            // index the ez component of the source
            *(sources[s].ez_x) = *(sources[s].ez_x) + (sources[s].values)[n];

            *(sources[s].ez_y) = *(sources[s].ez_y) + (sources[s].values)[n];

            *(sources[s].ez)  = *(sources[s].ez_x) + *(sources[s].ez_y);

            // put the resulting total voltage in the source_values array once the value is used for this
            // time step.
            (sources[s].values)[n] = *(sources[s].ez);
        }

        {
            // lock the mutex while updating shared variable, also ensures that only one thread sends a notification
            // to the controller at a time, preventing missed notifications.
            std::unique_lock<std::mutex> lock(mutex);
            ++e_updates;

            cv.notify_all(); 
            // wait for all threads to reach this point before moving on to h-field updates.
            cv_th.wait(lock, [] { return e_updates_done; });
        }
        
        solver_update_hx(x_start, x_stop);
        solver_update_hy(x_start, x_stop);
        solver_update_hz(x_start, x_stop);
        
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

            if ((n > 0) && ((n % (mon.n_step)) != 0))
            {
                continue;
            }

            int m_n = n / mon.n_step;

            MatrixFloatType m_values (mon.values + (m_n * (mon.N1N2)), mon.N1, mon.N2);
            
            if (mon.axis == 0)
            {
                // only update x slice monitor if it is within bounds of the thread
                if ((mon.position < x_start) || (mon.position >= x_stop))
                {
                    continue;
                }
                MatrixFloatType m_field (mon.field + (mon.position * (mon.NyNz)), mon.Ny, mon.Nz);
                m_values = m_field;
            }

            else if (mon.axis == 1)
            {
                // components on the edge of the grid that have +1 components from the grid will not be captured here,
                // but they are not updated and will be zero. 
                // as long as the monitor values were initialized in python to zero, this is a non-issue.
                for (int i = x_start; i < x_stop; i++)
                {
                    MatrixFloatType m_field (mon.field + (i * (mon.NyNz)), mon.Ny, mon.Nz);
                    m_values.row(i) = m_field.row(mon.position);
                }
            }

            else // (m.axis == 2)
            {
                for (int i = x_start; i < x_stop; i++)
                {
                    MatrixFloatType m_field (mon.field + (i * (mon.NyNz)), mon.Ny, mon.Nz);
                    m_values.row(i) = m_field.col(mon.position);
                }
            }
            
        }

    }

}

// update Ex components, starting with the x-axis index x_start, and ending with x_stop (stop index is not inclusive)
int solver_update_ex(int x_start, int x_stop)
{
    int x_offset;
    // number of updated field components
    int Ny = Cx.Ny;
    int Nz = Cx.Nz;

    // operate on a single slice of the field on the x axis
    for (int x = x_start; x < x_stop; x++)
    {   
        x_offset = x * Ex.NyNz;
        MatrixFloatType ex_y (Ex.ex_y + x_offset, Ex.Ny, Ex.Nz);
        MatrixFloatType ex_z (Ex.ex_z + x_offset, Ex.Ny, Ex.Nz);
        MatrixFloatType ex   (Ex.ex   + x_offset, Ex.Ny, Ex.Nz);
        
        // hz and hy are in the same x plane as ex
        MatrixFloatType hz (Hz.hz + (x * Hz.NyNz), Hz.Ny, Hz.Nz);
        MatrixFloatType hy (Hy.hy + (x * Hy.NyNz), Hy.Ny, Hy.Nz);
        
        // ex coefficients have the same x index as ex
        x_offset = x * Cx.NyNz;
        MatrixFloatType Cb_ex_y (Cx.Cb_ex_y + x_offset, Ny, Nz);
        MatrixFloatType Cb_ex_z (Cx.Cb_ex_z + x_offset, Ny, Nz);

        MatrixFloatType Ca_ex_y (Cx.Ca_ex_y + x_offset, Ny, Nz);
        MatrixFloatType Ca_ex_z (Cx.Ca_ex_z + x_offset, Ny, Nz);
        
        // update ex_y
        // ex_y[:, 1:-1, 1:-1] = (Ca_ex_y * ex_y[:, 1:-1, 1:-1]) + ex_yd
        ex_y.block(1, 1, Ny, Nz) = Ca_ex_y.cwiseProduct(ex_y.block(1, 1, Ny, Nz)) + (
            Cb_ex_y.cwiseProduct((hz.bottomRows(Ny) - hz.topRows(Ny)).block(0, 1, Ny, Nz))
        );
        
        // ex_yd = Cb_ex_y * np.diff(hz, axis=1)[:, :, 1:-1]
        // ex_y.block(1, 1, Ny, Nz) += (
        //     Cb_ex_y.cwiseProduct((hz.bottomRows(Ny) - hz.topRows(Ny)).block(0, 1, Ny, Nz))
        // );

        // update ex_z
        // ex_z[:, 1:-1, 1:-1] = (Ca_ex_z * ex_z[:, 1:-1, 1:-1]) + ex_zd
        ex_z.block(1, 1, Ny, Nz) = Ca_ex_z.cwiseProduct(ex_z.block(1, 1, Ny, Nz)) + (
            Cb_ex_z.cwiseProduct((hy.rightCols(Nz) - hy.leftCols(Nz)).block(1, 0, Ny, Nz))
        );

        // ex_zd = Cb_ex_z * np.diff(hy, axis=2)[:, 1:-1, :]
        // ex_z.block(1, 1, Ny, Nz) += (
        //     Cb_ex_z.cwiseProduct((hy.rightCols(Nz) - hy.leftCols(Nz)).block(1, 0, Ny, Nz))
        // );

        // combine split components
        ex = ex_y + ex_z;

    }

    return 0;
}

// update Ey components, starting with the x-axis index x_start, and ending with x_stop (stop index is not inclusive)
int solver_update_ey(int x_start, int x_stop)
{
    int x_offset;
    // number of updated field components
    int Ny = Cy.Ny;
    int Nz = Cy.Nz;

    // get a single slice of the field on the x axis
    for (int x = x_start; x < x_stop; x++)
    {   
        // endpoints of ey along x do not get updated
        if ((x < 1) || (x > (Ey.Nx - 2)))
        {
            continue;
        }
        
        x_offset = x * Ey.NyNz;
        MatrixFloatType ey_z (Ey.ey_z + x_offset, Ey.Ny, Ey.Nz);
        MatrixFloatType ey_x (Ey.ey_x + x_offset, Ey.Ny, Ey.Nz);
        MatrixFloatType ey   (Ey.ey   + x_offset, Ey.Ny, Ey.Nz);

        // hx is in the same x plane as ey
        MatrixFloatType hx (Hx.hx + (x * Hx.NyNz), Hx.Ny, Hx.Nz);
        
        // get hz components on either side of x-slice
        MatrixFloatType hz_0 (Hz.hz + ((x - 1) * Hz.NyNz), Hz.Ny, Hz.Nz);
        MatrixFloatType hz_1 (Hz.hz + (x * Hz.NyNz), Hz.Ny, Hz.Nz);

        // ey coefficients start at ex=1, the pointer offset for the first slice of coefficients at x=0 should be 0
        x_offset = (x - 1) * Cy.NyNz;
        MatrixFloatType Cb_ey_z (Cy.Cb_ey_z + x_offset, Ny, Nz);
        MatrixFloatType Cb_ey_x (Cy.Cb_ey_x + x_offset, Ny, Nz);

        MatrixFloatType Ca_ey_z (Cy.Ca_ey_z + x_offset, Ny, Nz);
        MatrixFloatType Ca_ey_x (Cy.Ca_ey_x + x_offset, Ny, Nz);

        // update ey_z
        // ey_z[1:-1, :, 1:-1] = (Ca_ey_z * ey_z[1:-1, :, 1:-1]) + ey_zd
        ey_z.block(0, 1, Ny, Nz) = Ca_ey_z.cwiseProduct(ey_z.block(0, 1, Ny, Nz)) + (
            Cb_ey_z.cwiseProduct(hx.rightCols(Nz) - hx.leftCols(Nz))
        );

        // ey_zd = Cb_ey_z * np.diff(hx, axis=2)[1:-1, :, :]
        // ey_z.block(0, 1, Ny, Nz) += (
        //     Cb_ey_z.cwiseProduct(hx.rightCols(Nz) - hx.leftCols(Nz))
        // );

        // update ey_x
        // ey_x[1:-1, :, 1:-1] = (Ca_ey_x * ey_x[1:-1, :, 1:-1]) + ey_xd
        ey_x.block(0, 1, Ny, Nz) = Ca_ey_x.cwiseProduct(ey_x.block(0, 1, Ny, Nz)) + (
            Cb_ey_x.cwiseProduct((hz_1 - hz_0).block(0, 1, Ny, Nz))
        );


        // ey_xd = Cb_ey_x * np.diff(hz, axis=0)[:, :, 1:-1]
        // ey_x.block(0, 1, Ny, Nz) += (
        //     Cb_ey_x.cwiseProduct((hz_1 - hz_0).block(0, 1, Ny, Nz))
        // );

        // combine split components
        ey = ey_z + ey_x;
    }

    return 0;
}


// update Ez components, starting with the x-axis index x_start, and ending with x_stop (stop index is not inclusive)
int solver_update_ez(int x_start, int x_stop)
{
    int x_offset;
    // number of updated field components
    int Ny = Cz.Ny;
    int Nz = Cz.Nz;

    // get a single slice of the field on the x axis
    for (int x = x_start; x < x_stop; x++)
    {   
        // endpoints of ey along x do not get updated
        if ((x < 1) || (x > (Ez.Nx - 2)))
        {
            continue;
        }
        
        x_offset = x * Ez.NyNz;
        MatrixFloatType ez_x (Ez.ez_x + x_offset, Ez.Ny, Ez.Nz);
        MatrixFloatType ez_y (Ez.ez_y + x_offset, Ez.Ny, Ez.Nz);
        MatrixFloatType ez   (Ez.ez   + x_offset, Ez.Ny, Ez.Nz);

        // hx is in the same x plane as ey
        MatrixFloatType hx (Hx.hx + (x * Hx.NyNz), Hx.Ny, Hx.Nz);

        // get hy components on either side of x-slice
        MatrixFloatType hy_0 (Hy.hy + ((x - 1) * Hy.NyNz), Hy.Ny, Hy.Nz);
        MatrixFloatType hy_1 (Hy.hy + (x * Hy.NyNz), Hy.Ny, Hy.Nz);

        // ez coefficients start at x=1, the pointer offset for the first slice of coefficients at x=0 should be 0
        x_offset = (x - 1) * Cz.NyNz;
        MatrixFloatType Cb_ez_x (Cz.Cb_ez_x + x_offset, Ny, Nz);
        MatrixFloatType Cb_ez_y (Cz.Cb_ez_y + x_offset, Ny, Nz);

        MatrixFloatType Ca_ez_x (Cz.Ca_ez_x + x_offset, Ny, Nz);
        MatrixFloatType Ca_ez_y (Cz.Ca_ez_y + x_offset, Ny, Nz);

        // update ez_x
        // ez_x[1:-1, 1:-1, :] = (Ca_ez_x * ez_x[1:-1, 1:-1, :]) + ez_xd
        ez_x.block(1, 0, Ny, Nz) = Ca_ez_x.cwiseProduct(ez_x.block(1, 0, Ny, Nz)) + (
            Cb_ez_x.cwiseProduct((hy_1 - hy_0).block(1, 0, Ny, Nz))
        );

        // ez_xd = Cb_ez_x * np.diff(hy, axis=0)[:, 1:-1, :]
        // ez_x.block(1, 0, Ny, Nz) += (
        //     Cb_ez_x.cwiseProduct((hy_1 - hy_0).block(1, 0, Ny, Nz))
        // );

        // update ez_y
        // ez_y[1:-1, 1:-1, :] = (Ca_ez_y * ez_y[1:-1, 1:-1, :]) + ez_yd
        ez_y.block(1, 0, Ny, Nz) = Ca_ez_y.cwiseProduct(ez_y.block(1, 0, Ny, Nz)) + (
            Cb_ez_y.cwiseProduct(hx.bottomRows(Ny) - hx.topRows(Ny))
        );

        // ez_yd = Cb_ez_y * np.diff(hx, axis=1)[1:-1, :, :]
        // ez_y.block(1, 0, Ny, Nz) += (
        //     Cb_ez_y.cwiseProduct(hx.bottomRows(Ny) - hx.topRows(Ny))
        // );

        // combine split components
        ez = ez_x + ez_y;

    }

    return 0;

}

// update Hx components, starting with the x-axis index x_start, and ending with x_stop (stop index is not inclusive)
int solver_update_hx(int x_start, int x_stop)
{
    int x_offset;
    // number of updated field components
    int Ny = Hx.Ny;
    int Nz = Hx.Nz;
    int NyNz = Hx.NyNz;

    // get a single slice of the field on the x axis
    for (int x = x_start; x < x_stop; x++)
    {   
        x_offset = x * NyNz;
        MatrixFloatType hx_y (Hx.hx_y + x_offset, Ny, Nz);
        MatrixFloatType hx_z (Hx.hx_z + x_offset, Ny, Nz);
        MatrixFloatType hx   (Hx.hx   + x_offset, Ny, Nz);

        // ey and ez are in the same x plane as hx
        MatrixFloatType ey (Ey.ey + (x * Ey.NyNz), Ey.Ny, Ey.Nz);
        MatrixFloatType ez (Ez.ez + (x * Ez.NyNz), Ez.Ny, Ez.Nz);

        // hx coefficients
        MatrixFloatType Db_hx_y (Dx.Db_hx_y + x_offset, Ny, Nz);
        MatrixFloatType Db_hx_z (Dx.Db_hx_z + x_offset, Ny, Nz);

        MatrixFloatType Da_hx_y (Dx.Da_hx_y + x_offset, Ny, Nz);
        MatrixFloatType Da_hx_z (Dx.Da_hx_z + x_offset, Ny, Nz);

        // update hx_y
        // hx_y = Da_hx_y * hx_y + hx_yd
        hx_y = Da_hx_y.cwiseProduct(hx_y) + (
            Db_hx_y.cwiseProduct(ez.bottomRows(Ny) - ez.topRows(Ny))
        );


        // hx_yd = Db_hx_y * np.diff(ez, axis=1)
        // hx_y += (
        //     Db_hx_y.cwiseProduct(ez.bottomRows(Ny) - ez.topRows(Ny))
        // );

        // update hx_z
        // hx_z = Da_hx_z * hx_z + hx_zd
        hx_z = Da_hx_z.cwiseProduct(hx_z) + (
            Db_hx_z.cwiseProduct(ey.rightCols(Nz) - ey.leftCols(Nz))
        );

        // hx_zd = Db_hx_z * np.diff(ey, axis=2)
        // hx_z += (
        //     Db_hx_z.cwiseProduct(ey.rightCols(Nz) - ey.leftCols(Nz))
        // );

        // combine split components
        hx = hx_y + hx_z;
    }

    return 0;

}

// update Hy components, starting with the x-axis index x_start, and ending with x_stop (stop index is not inclusive)
int solver_update_hy(int x_start, int x_stop)
{
    int x_offset;
    // number of updated field components
    int Ny = Hy.Ny;
    int Nz = Hy.Nz;
    int NyNz = Hy.NyNz;

    // get a single slice of the field on the x axis
    for (int x = x_start; x < x_stop; x++)
    {  
        x_offset = x * NyNz;
        MatrixFloatType hy_z (Hy.hy_z + x_offset, Ny, Nz);
        MatrixFloatType hy_x (Hy.hy_x + x_offset, Ny, Nz);
        MatrixFloatType hy   (Hy.hy   + x_offset, Ny, Nz);

        // ex is in the same x plane as hy
        MatrixFloatType ex (Ex.ex + (x * Ex.NyNz), Ex.Ny, Ex.Nz);

        // get ez components on either side of x-slice
        MatrixFloatType ez_0 (Ez.ez + ((x) * Ez.NyNz), Ez.Ny, Ez.Nz);
        MatrixFloatType ez_1 (Ez.ez + ((x + 1) * Ez.NyNz), Ez.Ny, Ez.Nz);

        // hy coefficients
        MatrixFloatType Db_hy_z (Dy.Db_hy_z + x_offset, Ny, Nz);
        MatrixFloatType Db_hy_x (Dy.Db_hy_x + x_offset, Ny, Nz);

        MatrixFloatType Da_hy_z (Dy.Da_hy_z + x_offset, Ny, Nz);
        MatrixFloatType Da_hy_x (Dy.Da_hy_x + x_offset, Ny, Nz);

        // update hy_z
        // hy_z = Da_hy_z * hy_z + hy_zd
        hy_z = Da_hy_z.cwiseProduct(hy_z) + (
            Db_hy_z.cwiseProduct(ex.rightCols(Nz) - ex.leftCols(Nz))
        );

        // hy_zd = Db_hy_z * np.diff(ex, axis=2)
        // hy_z += (
        //     Db_hy_z.cwiseProduct(ex.rightCols(Nz) - ex.leftCols(Nz))
        // );

        // update hy_x
        // hy_x = Da_hy_x * hy_x + hy_xd
        hy_x = Da_hy_x.cwiseProduct(hy_x) + (
            Db_hy_x.cwiseProduct(ez_1 - ez_0)
        );

        // hy_xd = Db_hy_x * np.diff(ez, axis=0)
        // hy_x += (
        //     Db_hy_x.cwiseProduct(ez_1 - ez_0)
        // );

        // combine split components
        hy = hy_z + hy_x;

    }

    return 0;
}

// update Hz components, starting with the x-axis index x_start, and ending with x_stop (stop index is not inclusive)
int solver_update_hz(int x_start, int x_stop)
{
    int x_offset;
    // number of updated field components
    int Ny = Hz.Ny;
    int Nz = Hz.Nz;
    int NyNz = Hz.NyNz;

    // get a single slice of the field on the x axis
    for (int x = x_start; x < x_stop; x++)
    {  
        x_offset = x * NyNz;
        MatrixFloatType hz_x (Hz.hz_x + x_offset, Ny, Nz);
        MatrixFloatType hz_y (Hz.hz_y + x_offset, Ny, Nz);
        MatrixFloatType hz   (Hz.hz   + x_offset, Ny, Nz);

        // ex is in the same x plane as hz
        MatrixFloatType ex (Ex.ex + (x * Ex.NyNz), Ex.Ny, Ex.Nz);

        // get ey components on either side of x-slice
        MatrixFloatType ey_0 (Ey.ey + ((x) * Ey.NyNz), Ey.Ny, Ey.Nz);
        MatrixFloatType ey_1 (Ey.ey + ((x + 1) * Ey.NyNz), Ey.Ny, Ey.Nz);

        // hz coefficients
        MatrixFloatType Db_hz_x (Dz.Db_hz_x + x_offset, Ny, Nz);
        MatrixFloatType Db_hz_y (Dz.Db_hz_y + x_offset, Ny, Nz);

        MatrixFloatType Da_hz_x (Dz.Da_hz_x + x_offset, Ny, Nz);
        MatrixFloatType Da_hz_y (Dz.Da_hz_y + x_offset, Ny, Nz);

        // update hz_x
        // hz_x = Da_hz_x * hz_x + hz_xd
        hz_x = Da_hz_x.cwiseProduct(hz_x) + (
            Db_hz_x.cwiseProduct(ey_1 - ey_0)
        );


        // hz_xd = Db_hz_x * np.diff(ey, axis=0) 
        // hz_x += (
        //     Db_hz_x.cwiseProduct(ey_1 - ey_0)
        // );

        // update hz_y
        // hz_y = Da_hz_y * hz_y + hz_yd
        hz_y = Da_hz_y.cwiseProduct(hz_y) + (
            Db_hz_y.cwiseProduct(ex.bottomRows(Ny) - ex.topRows(Ny))
        );

        // hz_yd = Db_hz_y * np.diff(ex, axis=1)
        // hz_y += (
        //     Db_hz_y.cwiseProduct(ex.bottomRows(Ny) - ex.topRows(Ny))
        // );

        // combine split components
        hz = hz_x + hz_y;
    }

    return 0;

}