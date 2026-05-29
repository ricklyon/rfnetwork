#include <iostream>

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

#include "solver.h"

#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Ex update kernel
__global__ void update_ex_y(float* Ca_ex_y, float* Cb_ex_y, float* ex_y, float* hz, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny - 1)) | (z_idx >= (Nz - 1)) )
    {
        return;
    }

    int x_stride = (x_idx * Ny * Nz);
    // grid indexing starts at the second ex component along y and z (edge components are not updated)
    int ex_idx  = x_stride + (y_idx * Nz) + z_idx;
    int hz1_idx = x_stride + ((y_idx) * Nz) + z_idx;
    int hz2_idx = x_stride + ((y_idx + 1) * Nz) + z_idx;

    ex_y[ex_idx] = Ca_ex_y[ex_idx] * ex_y[ex_idx] + Cb_ex_y[ex_idx] * (hz[hz2_idx] - hz[hz1_idx]);
}

__global__ void update_ex_z(float* Ca_ex_z , float* Cb_ex_z, float* ex_z, float* hy, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny - 1)) | (z_idx >= (Nz - 1)) )
    {
        return;
    }

    int x_stride = (x_idx * Ny * Nz);
    // grid indexing starts at the second ex component along y and z (edge components are not updated)
    int ex_idx  = x_stride + (y_idx * Nz) + z_idx;
    int hy1_idx = x_stride + (y_idx * Nz) + z_idx;
    int hy2_idx = x_stride + (y_idx * Nz) + z_idx + 1;

    ex_z[ex_idx] = Ca_ex_z[ex_idx] * ex_z[ex_idx] + Cb_ex_z[ex_idx] * (hy[hy2_idx] - hy[hy1_idx]);
}

// Ey update kernel
__global__ void update_ey_z(float* Ca_ey_z, float* Cb_ey_z, float* ey_z, float* hx, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx - 1)) | (y_idx >= (Ny)) | (z_idx >= (Nz - 1)))
    {
        return;
    }

    int x_stride = (x_idx * Ny * Nz);
    // grid indexing starts at the second ey component along x and z (edge components are not updated)
    int ey_idx  = x_stride + (y_idx * Nz) + z_idx;
    int hx1_idx = x_stride + (y_idx * Nz) + z_idx;
    int hx2_idx = x_stride + (y_idx * Nz) + z_idx + 1;

    ey_z[ey_idx] = Ca_ey_z[ey_idx] * ey_z[ey_idx] + Cb_ey_z[ey_idx] * (hx[hx2_idx] - hx[hx1_idx]);
}

__global__ void update_ey_x(float* Ca_ey_x, float* Cb_ey_x, float* ey_x, float* hz, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx - 1)) | (y_idx >= (Ny)) | (z_idx >= (Nz - 1)))
    {
        return;
    }

    // grid indexing starts at the second ey component along x and z (edge components are not updated)
    int ey_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hz1_idx = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hz2_idx = ((x_idx + 1) * Ny * Nz) + (y_idx * Nz) + z_idx;

    ey_x[ey_idx] = Ca_ey_x[ey_idx] * ey_x[ey_idx] + Cb_ey_x[ey_idx] * (hz[hz2_idx] - hz[hz1_idx]);
}

// Ez update kernel
__global__ void update_ez_x(float* Ca_ez_x, float* Cb_ez_x, float* ez_x, float* hy, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx - 1)) | (y_idx >= (Ny - 1)) | (z_idx >= (Nz)))
    {
        return;
    }

    // grid indexing starts at the second ey component along x and z (edge components are not updated)
    int ez_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hy1_idx = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hy2_idx = ((x_idx + 1) * Ny * Nz) + (y_idx * Nz) + z_idx;

    ez_x[ez_idx] = Ca_ez_x[ez_idx] * ez_x[ez_idx] + Cb_ez_x[ez_idx] * (hy[hy2_idx] - hy[hy1_idx]);
}

__global__ void update_ez_y(float* Ca_ez_y, float* Cb_ez_y, float* ez_y, float* hx, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx - 1)) | (y_idx >= (Ny - 1)) | (z_idx >= (Nz)))
    {
        return;
    }

    // grid indexing starts at the second ey component along x and z (edge components are not updated)
    int ez_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hx1_idx = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hx2_idx = (x_idx * Ny * Nz) + ((y_idx + 1) * Nz) + z_idx;

    ez_y[ez_idx] = Ca_ez_y[ez_idx] * ez_y[ez_idx] + Cb_ez_y[ez_idx] * (hx[hx2_idx] - hx[hx1_idx]);
}

// Add split e-field components
__global__ void combine_ex(float* ex , float* ex_y, float* ex_z, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny - 1)) | (z_idx >= (Nz - 1)) )
    {
        return;
    }

    int ex_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    ex[ex_idx] = ex_y[ex_idx] + ex_z[ex_idx];
}

__global__ void combine_ey(float* ey , float* ey_z, float* ey_x, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx - 1)) | (y_idx >= (Ny)) | (z_idx >= (Nz - 1)))
    {
        return;
    }

    int ey_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    ey[ey_idx] = ey_z[ey_idx] + ey_x[ey_idx];
}

__global__ void combine_ez(float* ez , float* ez_x, float* ez_y, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx - 1)) | (y_idx >= (Ny - 1)) | (z_idx >= (Nz)))
    {
        return;
    }

    int ez_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    ez[ez_idx] = ez_x[ez_idx] + ez_y[ez_idx];
}

// Hx update kernel
__global__ void update_hx_y(float* Da_hx_y, float* Db_hx_y1, float * Db_hx_y2, float* hx_y, float* ez, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx - 1)) | (y_idx >= (Ny)) | (z_idx >= (Nz)) )
    {
        return;
    }

    int x_stride = (x_idx * Ny * Nz);
    // grid indexing starts at the second hx component along x (edge components are not updated)
    int hx_idx  = x_stride + (y_idx * Nz) + z_idx;
    int ez1_idx = x_stride + ((y_idx - 1) * Nz) + z_idx;
    int ez2_idx = x_stride + ((y_idx) * Nz) + z_idx;

    // use ez=0 if referencing the ez component at the edge of the grid
    float ez1 = y_idx > 0 ? ez[ez1_idx] : 0;

    hx_y[hx_idx] = Da_hx_y[hx_idx] * hx_y[hx_idx] + (Db_hx_y2[hx_idx] * ez[ez2_idx]) - (Db_hx_y1[hx_idx] * ez1);
}

__global__ void update_hx_z(float* Da_hx_z, float* Db_hx_z1, float * Db_hx_z2, float* hx_z, float* ey, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx - 1)) | (y_idx >= (Ny)) | (z_idx >= (Nz)) )
    {
        return;
    }

    int x_stride = (x_idx * Ny * Nz);
    // grid indexing starts at the second hx component along x (edge components are not updated)
    int hx_idx  = x_stride + (y_idx * Nz) + z_idx;
    int ey1_idx = x_stride + ((y_idx) * Nz) + z_idx - 1;
    int ey2_idx = x_stride + ((y_idx) * Nz) + z_idx;

    // use ez=0 if referencing the ez component at the edge of the grid
    float ey1 = z_idx > 0 ? ey[ey1_idx] : 0;

    hx_z[hx_idx] = Da_hx_z[hx_idx] * hx_z[hx_idx] + (Db_hx_z2[hx_idx] * ey[ey2_idx]) - (Db_hx_z1[hx_idx] * ey1);
}

// Hy update kernel
__global__ void update_hy_z(float* Da_hy_z, float* Db_hy_z1, float * Db_hy_z2, float* hy_z, float* ex, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny - 1)) | (z_idx >= (Nz)) )
    {
        return;
    }

    int x_stride = (x_idx * Ny * Nz);
    // grid indexing starts at the second hy component along y (edge components are not updated)
    int hy_idx  = x_stride + (y_idx * Nz) + z_idx;
    int ex1_idx = x_stride + ((y_idx) * Nz) + z_idx - 1;
    int ex2_idx = x_stride + ((y_idx) * Nz) + z_idx;

    // use ex=0 if referencing the ex component at the edge of the grid
    float ex1 = z_idx > 0 ? ex[ex1_idx] : 0;

    hy_z[hy_idx] = Da_hy_z[hy_idx] * hy_z[hy_idx] + (Db_hy_z2[hy_idx] * ex[ex2_idx]) - (Db_hy_z1[hy_idx] * ex1);
}

__global__ void update_hy_x(float* Da_hy_x, float* Db_hy_x1, float * Db_hy_x2, float* hy_x, float* ez, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny - 1)) | (z_idx >= (Nz)) )
    {
        return;
    }

    // grid indexing starts at the second hy component along y (edge components are not updated)
    int hy_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int ez1_idx = ((x_idx - 1) * Ny * Nz) + ((y_idx) * Nz) + z_idx;
    int ez2_idx = (x_idx * Ny * Nz) + ((y_idx) * Nz) + z_idx;

    // use ex=0 if referencing the ex component at the edge of the grid
    float ez1 = x_idx > 0 ? ez[ez1_idx] : 0;

    hy_x[hy_idx] = Da_hy_x[hy_idx] * hy_x[hy_idx] + (Db_hy_x2[hy_idx] * ez[ez2_idx]) - (Db_hy_x1[hy_idx] * ez1);
}

__global__ void update_hz_x(float* Da_hz_x, float* Db_hz_x1, float * Db_hz_x2, float* hz_x, float* ey, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny)) | (z_idx >= (Nz - 1)) )
    {
        return;
    }

    // grid indexing starts at the second hz component along z (edge components are not updated)
    int hx_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int ey1_idx = ((x_idx - 1) * Ny * Nz) + ((y_idx) * Nz) + z_idx;
    int ey2_idx = (x_idx * Ny * Nz) + ((y_idx) * Nz) + z_idx;

    // use ey=0 if referencing the ex component at the edge of the grid
    float ey1 = x_idx > 0 ? ey[ey1_idx] : 0;

    hz_x[hx_idx] = Da_hz_x[hx_idx] * hz_x[hx_idx] + (Db_hz_x2[hx_idx] * ey[ey2_idx]) - (Db_hz_x1[hx_idx] * ey1);
}

__global__ void update_hz_y(float* Da_hz_y, float* Db_hz_y1, float * Db_hz_y2, float* hz_y, float* ex, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny)) | (z_idx >= (Nz - 1)) )
    {
        return;
    }

    // grid indexing starts at the second hz component along z (edge components are not updated)
    int x_stride = (x_idx * Ny * Nz);
    int hz_idx  = x_stride + (y_idx * Nz) + z_idx;
    int ex1_idx = x_stride + ((y_idx - 1) * Nz) + z_idx;
    int ex2_idx = x_stride + ((y_idx) * Nz) + z_idx;

    // use ey=0 if referencing the ex component at the edge of the grid
    float ex1 = y_idx > 0 ? ex[ex1_idx] : 0;

    hz_y[hz_idx] = Da_hz_y[hz_idx] * hz_y[hz_idx] + (Db_hz_y2[hz_idx] * ex[ex2_idx]) - (Db_hz_y1[hz_idx] * ex1);
}

// Add split e-field components
__global__ void combine_hx(float* hx, float* hx_y, float* hx_z, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx - 1)) | (y_idx >= (Ny)) | (z_idx >= (Nz)) )
    {
        return;
    }

    int hx_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    hx[hx_idx] = hx_y[hx_idx] + hx_z[hx_idx];
}

__global__ void combine_hy(float* hy, float* hy_z, float* hy_x, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny - 1)) | (z_idx >= (Nz)))
    {
        return;
    }

    int hy_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    hy[hy_idx] = hy_z[hy_idx] + hy_x[hy_idx];
}

__global__ void combine_hz(float* hz, float* hz_x, float* hz_y, int Nx, int Ny, int Nz)
{
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is on or after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny)) | (z_idx >= (Nz - 1)))
    {
        return;
    }

    int hz_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    hz[hz_idx] = hz_x[hz_idx] + hz_y[hz_idx];
}

__global__ void update_source(float* field, float* field_sp1, float* field_sp2, float value)
{
    field_sp1[0] = field_sp1[0] + value;
    field_sp2[0] = field_sp2[0] + value;
    field[0] = field_sp1[0] + field_sp2[0];
}

__global__ void field_update_kernel(
    float* Ca_ex_y, float* Cb_ex_y, float* ex_y,
    float* Ca_ex_z , float* Cb_ex_z, float* ex_z,
    float* Ca_ey_z, float* Cb_ey_z, float* ey_z,
    float* Ca_ey_x, float* Cb_ey_x, float* ey_x,
    float* Ca_ez_x, float* Cb_ez_x, float* ez_x,
    float* Ca_ez_y, float* Cb_ez_y, float* ez_y,
    float* Da_hx_y, float* Db_hx_y1, float * Db_hx_y2, float* hx_y,
    float* Da_hx_z, float* Db_hx_z1, float * Db_hx_z2, float* hx_z,
    float* Da_hy_z, float* Db_hy_z1, float * Db_hy_z2, float* hy_z,
    float* Da_hy_x, float* Db_hy_x1, float * Db_hy_x2, float* hy_x,
    float* Da_hz_x, float* Db_hz_x1, float * Db_hz_x2, float* hz_x,
    float* Da_hz_y, float* Db_hz_y1, float * Db_hz_y2, float* hz_y,
    float* ex, float* ey, float* ez, float* hx, float* hy, float* hz,
    int* probe_idx, int* probe_type, int* probe_is_src, float* probe_values, 
    int Nx, int Ny, int Nz, int Nt, int Np
)
{

    cg::grid_group grid = cg::this_grid();

    int x_idx = threadIdx.z + blockIdx.z * blockDim.z;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.x + blockIdx.x * blockDim.x;

    // skip update if thread is after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny)) | (z_idx >= (Nz)) )
    {
        return;
    }

    // field index, applies to e and h fields
    int f_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;

    // number of probes in this cell (up to 6 if all components have probes, only 3 are supported right now)
    int n_probe = 0;
    int i_probe[3];
    float * p_probe[3];

    for (int i = 0; i < Np; i++)
    {   
        if ((probe_idx[i] == f_idx) && (n_probe < 3))
        {
            i_probe[n_probe] = i;

            if (probe_type[i] == 0){
                p_probe[n_probe] = ex;
            }
            else if (probe_type[i] == 1){
                p_probe[n_probe] = ey;
            }
            else if (probe_type[i] == 2){
                p_probe[n_probe] = ez;
            }
            else if (probe_type[i] == 3){
                p_probe[n_probe] = hx;
            }
            else if (probe_type[i] == 4){
                p_probe[n_probe] = hy;
            }
            else if (probe_type[i] == 5){
                p_probe[n_probe] = hz;
            }
            
            n_probe += 1;
        }
    }

    // h-field indices on either side of e-fields
    int hz_y2_idx = (x_idx * Ny * Nz) + ((y_idx + 1) * Nz) + z_idx;
    int hy_z2_idx = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx + 1;

    int hx_z2_idx = (x_idx * Ny * Nz)  + (y_idx * Nz) + z_idx + 1;
    int hz_x2_idx = ((x_idx + 1) * Ny * Nz) + (y_idx * Nz) + z_idx;

    int hy_x2_idx = ((x_idx + 1) * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hx_y2_idx = (x_idx * Ny * Nz) + ((y_idx + 1) * Nz) + z_idx;

    // e-field indices on either side of h-fields
    int ez_y1_idx = (x_idx * Ny * Nz) + ((y_idx - 1) * Nz) + z_idx;
    float ez_y1 = 0;

    int ey_z1_idx = (x_idx * Ny * Nz) + ((y_idx) * Nz) + z_idx - 1;
    float ey_z1 = 0;

    int ex_z1_idx = (x_idx * Ny * Nz) + ((y_idx) * Nz) + z_idx - 1;
    float ex_z1 = 0;

    int ez_x1_idx = ((x_idx - 1) * Ny * Nz) + ((y_idx) * Nz) + z_idx;
    float ez_x1 = 0;

    int ey_x1_idx = ((x_idx - 1) * Ny * Nz) + ((y_idx) * Nz) + z_idx;
    float ey_x1 = 0;

    int ex_y1_idx = (x_idx * Ny * Nz) + ((y_idx - 1) * Nz) + z_idx;
    float ex_y1 = 0;

    int p_idx;

    // main time stepping loop
    for (int n = 0; n < Nt; n++)
    {
        // update ex_y
        if (hz_y2_idx < Ny)
        {
            ex_y[f_idx] = Ca_ex_y[f_idx] * ex_y[f_idx] + Cb_ex_y[f_idx] * (hz[hz_y2_idx] - hz[f_idx]);
        }
        
        // update ex_z
        if (hy_z2_idx < Nz)
        {
            ex_z[f_idx] = Ca_ex_z[f_idx] * ex_z[f_idx] + Cb_ex_z[f_idx] * (hy[hy_z2_idx] - hy[f_idx]);
        }

        // update ey_z
        if (hx_z2_idx < Nz)
        {
            ey_z[f_idx] = Ca_ey_z[f_idx] * ey_z[f_idx] + Cb_ey_z[f_idx] * (hx[hx_z2_idx] - hx[f_idx]);
        }

        // update ey_x
        if (hz_x2_idx < Nx)
        {
            ey_x[f_idx] = Ca_ey_x[f_idx] * ey_x[f_idx] + Cb_ey_x[f_idx] * (hz[hz_x2_idx] - hz[f_idx]);
        }

        // update ez_x
        if (hy_x2_idx < Nx)
        {
            ez_x[f_idx] = Ca_ez_x[f_idx] * ez_x[f_idx] + Cb_ez_x[f_idx] * (hy[hy_x2_idx] - hy[f_idx]);
        }
        
        //update ez_y
        if (hx_y2_idx < Ny)
        {
            ez_y[f_idx] = Ca_ez_y[f_idx] * ez_y[f_idx] + Cb_ez_y[f_idx] * (hx[hx_y2_idx] - hx[f_idx]);
        }

        // combine E fields
        ex[f_idx] = ex_y[f_idx] + ex_z[f_idx];
        ey[f_idx] = ey_z[f_idx] + ey_x[f_idx];
        ez[f_idx] = ez_x[f_idx] + ez_y[f_idx];

        // cudaDeviceSynchronize();
        grid.sync();

        // update hx_y
        if (y_idx > 0)
        {
            ez_y1 = ez[ez_y1_idx];
        }
        hx_y[f_idx] = Da_hx_y[f_idx] * hx_y[f_idx] + (Db_hx_y2[f_idx] * ez[f_idx]) - (Db_hx_y1[f_idx] * ez_y1);

        // update hx_z
        if (z_idx > 0)
        {
            ey_z1 = ey[ey_z1_idx];
        }
        hx_z[f_idx] = Da_hx_z[f_idx] * hx_z[f_idx] + (Db_hx_z2[f_idx] * ey[f_idx]) - (Db_hx_z1[f_idx] * ey_z1);

        // update hy_z
        if (z_idx > 0)
        {
            ex_z1 = ex[ex_z1_idx];
        }
        hy_z[f_idx] = Da_hy_z[f_idx] * hy_z[f_idx] + (Db_hy_z2[f_idx] * ex[f_idx]) - (Db_hy_z1[f_idx] * ex_z1);

        // update hy_x
        if (z_idx > 0)
        {
            ez_x1 = ez[ez_x1_idx];
        }
        hy_x[f_idx] = Da_hy_x[f_idx] * hy_x[f_idx] + (Db_hy_x2[f_idx] * ez[f_idx]) - (Db_hy_x1[f_idx] * ez_x1);

        // update hz_x
        if (x_idx > 0){
            ey_x1 = ey[ey_x1_idx];
        }
        hz_x[f_idx] = Da_hz_x[f_idx] * hz_x[f_idx] + (Db_hz_x2[f_idx] * ey[f_idx]) - (Db_hz_x1[f_idx] * ey_x1);

        // update hz_y
        if (y_idx > 0)
        {
            ex_y1 = ex[ex_y1_idx];
        }
        hz_y[f_idx] = Da_hz_y[f_idx] * hz_y[f_idx] + (Db_hz_y2[f_idx] * ex[f_idx]) - (Db_hz_y1[f_idx] * ex_y1);

        // combine h-fields
        hx[f_idx] = hx_y[f_idx] + hx_z[f_idx];
        hy[f_idx] = hy_z[f_idx] + hy_x[f_idx];
        hz[f_idx] = hz_x[f_idx] + hz_y[f_idx];

        // Wait for the kernel to complete execution
        // cudaDeviceSynchronize();

        for (int i = 0; i < n_probe; i++)
        {   
            p_idx = i_probe[i];

            if (probe_is_src[p_idx])
            {
                if (probe_type[p_idx] == 0)
                {
                    ex_y[f_idx] += probe_values[(p_idx * Nt) + n];
                    ex_z[f_idx] += probe_values[(p_idx * Nt) + n];
                    ex[f_idx] = ex_y[f_idx] + ex_z[f_idx];
                }
                else if (probe_type[p_idx] == 1)
                {
                    ey_z[f_idx] += probe_values[(p_idx * Nt) + n];
                    ey_x[f_idx] += probe_values[(p_idx * Nt) + n];
                    ey[f_idx] = ey_z[f_idx] + ey_x[f_idx];
                }
                else if (probe_type[p_idx] == 2)
                {
                    ez_x[f_idx] += probe_values[(p_idx * Nt) + n];
                    ez_y[f_idx] += probe_values[(p_idx * Nt) + n];
                    ez[f_idx] = ez_x[f_idx] + ez_y[f_idx];
                }
                else if (probe_type[p_idx] == 3)
                {
                    hx_y[f_idx] += probe_values[(p_idx * Nt) + n];
                    hx_z[f_idx] += probe_values[(p_idx * Nt) + n];
                    hx[f_idx] = hx_y[f_idx] + hx_z[f_idx];
                }
                else if (probe_type[p_idx] == 4)
                {
                    hy_z[f_idx] += probe_values[(p_idx * Nt) + n];
                    hy_x[f_idx] += probe_values[(p_idx * Nt) + n];
                    hy[f_idx] = hy_z[f_idx] + hy_x[f_idx];
                }
                else if (probe_type[p_idx] == 5)
                {
                    hz_x[f_idx] += probe_values[(p_idx * Nt) + n];
                    hz_y[f_idx] += probe_values[(p_idx * Nt) + n];
                    hz[f_idx] = hz_x[f_idx] + hz_y[f_idx];
                }
            }

            // update probes
            // if this is a source, the values array is replaced with the resulting total voltage after it is used for
            // each time step.
            probe_values[(p_idx * Nt) + n] = p_probe[i][f_idx];
        }

        grid.sync();

    }
}


void SolverFDTD::solver_run_cu(int Nt)
{

    // number of total cells in each direction
    int Nx = Ex.Nx;
    int Ny = Ey.Ny;
    int Nz = Ez.Nz;

    // size of thread blocks in each direction
    int Nx_th = 8;
    int Ny_th = 8;
    int Nz_th = 8;

    // number of blocks in the grid
    int Nx_b = (Nx + Nx_th - 1) / Nx_th;
    int Ny_b = (Ny + Ny_th - 1) / Ny_th;
    int Nz_b = (Nz + Nz_th - 1) / Nz_th;

    // size of each field component grid
    int f_size = Nx * Ny * Nz * sizeof(float);

    // pointers to field component grids on device
    float * p_ex_y = nullptr;
    float * p_ex_z = nullptr;
    float * p_ex   = nullptr;

    cudaMalloc(&p_ex_y, f_size);
    cudaMalloc(&p_ex_z, f_size);
    cudaMalloc(&p_ex, f_size);

    float * p_ey_z = nullptr;
    float * p_ey_x = nullptr;
    float * p_ey   = nullptr;

    cudaMalloc(&p_ey_z, f_size);
    cudaMalloc(&p_ey_x, f_size);
    cudaMalloc(&p_ey, f_size);

    float * p_ez_x = nullptr;
    float * p_ez_y = nullptr;
    float * p_ez   = nullptr;

    cudaMalloc(&p_ez_x, f_size);
    cudaMalloc(&p_ez_y, f_size);
    cudaMalloc(&p_ez, f_size);

    float * p_hx_y = nullptr;
    float * p_hx_z = nullptr;
    float * p_hx   = nullptr;

    cudaMalloc(&p_hx_y, f_size);
    cudaMalloc(&p_hx_z, f_size);
    cudaMalloc(&p_hx, f_size);

    float * p_hy_z = nullptr;
    float * p_hy_x = nullptr;
    float * p_hy   = nullptr;

    cudaMalloc(&p_hy_z, f_size);
    cudaMalloc(&p_hy_x, f_size);
    cudaMalloc(&p_hy, f_size);

    float * p_hz_x = nullptr;
    float * p_hz_y = nullptr;
    float * p_hz   = nullptr;

    cudaMalloc(&p_hz_x, f_size);
    cudaMalloc(&p_hz_y, f_size);
    cudaMalloc(&p_hz, f_size);

    // pointers to e field coefficient arrays on device
    float * Ca_ex_y = nullptr;
    float * Ca_ex_z = nullptr;
    float * Cb_ex_y = nullptr;
    float * Cb_ex_z = nullptr;

    cudaMalloc(&Ca_ex_y, f_size);
    cudaMalloc(&Ca_ex_z, f_size);
    cudaMalloc(&Cb_ex_y, f_size);
    cudaMalloc(&Cb_ex_z, f_size);

    float * Ca_ey_z = nullptr;
    float * Ca_ey_x = nullptr;
    float * Cb_ey_z = nullptr;
    float * Cb_ey_x = nullptr;

    cudaMalloc(&Ca_ey_z, f_size);
    cudaMalloc(&Ca_ey_x, f_size);
    cudaMalloc(&Cb_ey_z, f_size);
    cudaMalloc(&Cb_ey_x, f_size);

    float * Ca_ez_x = nullptr;
    float * Ca_ez_y = nullptr;
    float * Cb_ez_x = nullptr;
    float * Cb_ez_y = nullptr;

    cudaMalloc(&Ca_ez_x, f_size);
    cudaMalloc(&Ca_ez_y, f_size);
    cudaMalloc(&Cb_ez_x, f_size);
    cudaMalloc(&Cb_ez_y, f_size);

    // pointers to h field coefficient arrays on device
    // hx coefficients
    float * Db_hx_y1 = nullptr;
    float * Db_hx_y2 = nullptr;
    float * Db_hx_z1 = nullptr;
    float * Db_hx_z2 = nullptr;

    float * Da_hx_y = nullptr;
    float * Da_hx_z = nullptr;

    cudaMalloc(&Db_hx_y1, f_size);
    cudaMalloc(&Db_hx_y2, f_size);
    cudaMalloc(&Db_hx_z1, f_size);
    cudaMalloc(&Db_hx_z2, f_size);

    cudaMalloc(&Da_hx_y, f_size);
    cudaMalloc(&Da_hx_z, f_size);

    // hy coefficients
    float * Db_hy_z1 = nullptr;
    float * Db_hy_z2 = nullptr;
    float * Db_hy_x1 = nullptr;
    float * Db_hy_x2 = nullptr;

    float * Da_hy_z = nullptr;
    float * Da_hy_x = nullptr;

    cudaMalloc(&Db_hy_z1, f_size);
    cudaMalloc(&Db_hy_z2, f_size);
    cudaMalloc(&Db_hy_x1, f_size);
    cudaMalloc(&Db_hy_x2, f_size);

    cudaMalloc(&Da_hy_z, f_size);
    cudaMalloc(&Da_hy_x, f_size);

    // hz coefficients
    float * Db_hz_x1 = nullptr;
    float * Db_hz_x2 = nullptr;
    float * Db_hz_y1 = nullptr;
    float * Db_hz_y2 = nullptr;

    float * Da_hz_x = nullptr;
    float * Da_hz_y = nullptr;

    cudaMalloc(&Db_hz_x1, f_size);
    cudaMalloc(&Db_hz_x2, f_size);
    cudaMalloc(&Db_hz_y1, f_size);
    cudaMalloc(&Db_hz_y2, f_size);

    cudaMalloc(&Da_hz_x, f_size);
    cudaMalloc(&Da_hz_y, f_size);

    // initialize fields to zero
    cudaMemset(p_ex_y, 0, f_size);
    cudaMemset(p_ex_z, 0, f_size);
    cudaMemset(p_ex, 0, f_size);

    cudaMemset(p_ey_z, 0, f_size);
    cudaMemset(p_ey_x, 0, f_size);
    cudaMemset(p_ey, 0, f_size);

    cudaMemset(p_ez_x, 0, f_size);
    cudaMemset(p_ez_y, 0, f_size);
    cudaMemset(p_ez, 0, f_size);

    cudaMemset(p_hx_y, 0, f_size);
    cudaMemset(p_hx_z, 0, f_size);
    cudaMemset(p_hx, 0, f_size);

    cudaMemset(p_hy_z, 0, f_size);
    cudaMemset(p_hy_x, 0, f_size);
    cudaMemset(p_hy, 0, f_size);

    cudaMemset(p_hz_x, 0, f_size);
    cudaMemset(p_hz_y, 0, f_size);
    cudaMemset(p_hz, 0, f_size);

    // copy coefficient arrays to GPU
    // e-fields
    cudaMemcpy(Ca_ex_y, Cx.Ca_ex_y, f_size, cudaMemcpyDefault);
    cudaMemcpy(Ca_ex_z, Cx.Ca_ex_z, f_size, cudaMemcpyDefault);

    cudaMemcpy(Ca_ey_z, Cy.Ca_ey_z, f_size, cudaMemcpyDefault);
    cudaMemcpy(Ca_ey_x, Cy.Ca_ey_x, f_size, cudaMemcpyDefault);

    cudaMemcpy(Ca_ez_x, Cz.Ca_ez_x, f_size, cudaMemcpyDefault);
    cudaMemcpy(Ca_ez_y, Cz.Ca_ez_y, f_size, cudaMemcpyDefault);

    cudaMemcpy(Cb_ex_y, Cx.Cb_ex_y, f_size, cudaMemcpyDefault);
    cudaMemcpy(Cb_ex_z, Cx.Cb_ex_z, f_size, cudaMemcpyDefault);

    cudaMemcpy(Cb_ey_z, Cy.Cb_ey_z, f_size, cudaMemcpyDefault);
    cudaMemcpy(Cb_ey_x, Cy.Cb_ey_x, f_size, cudaMemcpyDefault);

    cudaMemcpy(Cb_ez_x, Cz.Cb_ez_x, f_size, cudaMemcpyDefault);
    cudaMemcpy(Cb_ez_y, Cz.Cb_ez_y, f_size, cudaMemcpyDefault);

    // h-fields
    // a
    cudaMemcpy(Da_hx_y, Dx.Da_hx_y, f_size, cudaMemcpyDefault);
    cudaMemcpy(Da_hx_z, Dx.Da_hx_z, f_size, cudaMemcpyDefault);

    cudaMemcpy(Da_hy_z, Dy.Da_hy_z, f_size, cudaMemcpyDefault);
    cudaMemcpy(Da_hy_x, Dy.Da_hy_x, f_size, cudaMemcpyDefault);

    cudaMemcpy(Da_hz_x, Dz.Da_hz_x, f_size, cudaMemcpyDefault);
    cudaMemcpy(Da_hz_y, Dz.Da_hz_y, f_size, cudaMemcpyDefault);

    // b, hx
    cudaMemcpy(Db_hx_y1, Dx.Db_hx_y1, f_size, cudaMemcpyDefault);
    cudaMemcpy(Db_hx_y2, Dx.Db_hx_y2, f_size, cudaMemcpyDefault);

    cudaMemcpy(Db_hx_z1, Dx.Db_hx_z1, f_size, cudaMemcpyDefault);
    cudaMemcpy(Db_hx_z2, Dx.Db_hx_z2, f_size, cudaMemcpyDefault);

    // b, hy
    cudaMemcpy(Db_hy_z1, Dy.Db_hy_z1, f_size, cudaMemcpyDefault);
    cudaMemcpy(Db_hy_z2, Dy.Db_hy_z2, f_size, cudaMemcpyDefault);

    cudaMemcpy(Db_hy_x1, Dy.Db_hy_x1, f_size, cudaMemcpyDefault);
    cudaMemcpy(Db_hy_x2, Dy.Db_hy_x2, f_size, cudaMemcpyDefault);

    // b, hz
    cudaMemcpy(Db_hz_x1, Dz.Db_hz_x1, f_size, cudaMemcpyDefault);
    cudaMemcpy(Db_hz_x2, Dz.Db_hz_x2, f_size, cudaMemcpyDefault);

    cudaMemcpy(Db_hz_y1, Dz.Db_hz_y1, f_size, cudaMemcpyDefault);
    cudaMemcpy(Db_hz_y2, Dz.Db_hz_y2, f_size, cudaMemcpyDefault);

    int probe_idx[MAX_PROBES];
    int probe_type[MAX_PROBES];
    float * probe_values[MAX_PROBES];
    int probe_is_src[MAX_PROBES];

    int * probe_idx_dev = nullptr;
    int * probe_type_dev = nullptr;
    float * probe_values_dev = nullptr;
    int * probe_is_src_dev = nullptr;

    int probe_size = n_probes * sizeof(int);
    cudaMalloc(&probe_idx_dev, n_probes * sizeof(int));
    cudaMalloc(&probe_type_dev, n_probes * sizeof(int));
    cudaMalloc(&probe_values_dev, n_probes * Nt * sizeof(float));
    cudaMalloc(&probe_is_src_dev, n_probes * sizeof(int));

    for (int i = 0; i < n_probes; i++)
    {   
        Probe * p = &(probes[i]);

        probe_idx[i] = ((p->x_cell) * Nx) + ((p->y_cell) * Ny) + ((p->z_cell) * Nz);
        probe_type[i] = p->field_type;
        probe_is_src[i] = p->is_source;

        cudaMemcpy(probe_values_dev + (i * Nt), p->values, Nt * sizeof(float), cudaMemcpyDefault);
    }

    cudaMemcpy(probe_idx_dev, probe_idx, n_probes * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(probe_type_dev, probe_type, n_probes * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(probe_is_src_dev, probe_is_src, n_probes * sizeof(int), cudaMemcpyDefault);

    dim3 block_size(Nx_th, Ny_th, Nz_th);
    dim3 grid_size(Nx_b, Ny_b, Nz_b);

    void* args[] = { 
        &Ca_ex_y, &Cb_ex_y, &p_ex_y,
        &Ca_ex_z, &Cb_ex_z, &p_ex_z,
        &Ca_ey_z, &Cb_ey_z, &p_ey_z,
        &Ca_ey_x, &Cb_ey_x, &p_ey_x,
        &Ca_ez_x, &Cb_ez_x, &p_ez_x,
        &Ca_ez_y, &Cb_ez_y, &p_ez_y,
        &Da_hx_y, &Db_hx_y1, &Db_hx_y2, &p_hx_y,
        &Da_hx_z, &Db_hx_z1, &Db_hx_z2, &p_hx_z,
        &Da_hy_z, &Db_hy_z1, &Db_hy_z2, &p_hy_z,
        &Da_hy_x, &Db_hy_x1, &Db_hy_x2, &p_hy_x,
        &Da_hz_x, &Db_hz_x1, &Db_hz_x2, &p_hz_x,
        &Da_hz_y, &Db_hz_y1, &Db_hz_y2, &p_hz_y,
        &p_ex, &p_ey, &p_ez, &p_hx, &p_hy, &p_hz,
        &probe_idx_dev, &probe_type_dev, &probe_is_src_dev, &probe_values_dev, 
        &Nx, &Ny, &Nz, &Nt, &n_probes
    };

    cudaLaunchCooperativeKernel(
        field_update_kernel, grid_size, block_size, args
    );

    // copy probe values from device to CPU
    for (int i = 0; i < n_probes; i++)
    {   
        Probe * p = &(probes[i]);

        cudaMemcpy(p->values, probe_values_dev + (i * Nt), Nt * sizeof(float), cudaMemcpyDefault);
    }

    // field_update_kernel<<<grid_size, block_size>>>(
        // Ca_ex_y, Cb_ex_y, p_ex_y,
        // Ca_ex_z, Cb_ex_z, p_ex_z,
        // Ca_ey_z, Cb_ey_z, p_ey_z,
        // Ca_ey_x, Cb_ey_x, p_ey_x,
        // Ca_ez_x, Cb_ez_x, p_ez_x,
        // Ca_ez_y, Cb_ez_y, p_ez_y,
        // Da_hx_y, Db_hx_y1, Db_hx_y2, p_hx_y,
        // Da_hx_z, Db_hx_z1, Db_hx_z2, p_hx_z,
        // Da_hy_z, Db_hy_z1, Db_hy_z2, p_hy_z,
        // Da_hy_x, Db_hy_x1, Db_hy_x2, p_hy_x,
        // Da_hz_x, Db_hz_x1, Db_hz_x2, p_hz_x,
        // Da_hz_y, Db_hz_y1, Db_hz_y2, p_hz_y,
        // p_ex, p_ey, p_ez, p_hx, p_hy, p_hz,
        // probe_idx_dev, probe_type_dev, probe_is_src_dev, probe_values_dev, 
        // Nx, Ny, Nz, Nt, n_probes
    // );

    // // main time stepping loop
    // for (int n = 0; n < Nt; n++)
    // {
    //     // update e-fields
    //     update_ex_y<<<grid_size, block_size>>>(Ca_ex_y, Cb_ex_y, p_ex_y, p_hz, Nx, Ny, Nz);
    //     update_ex_z<<<grid_size, block_size>>>(Ca_ex_z, Cb_ex_z, p_ex_z, p_hy, Nx, Ny, Nz);

    //     update_ey_z<<<grid_size, block_size>>>(Ca_ey_z, Cb_ey_z, p_ey_z, p_hx, Nx, Ny, Nz);
    //     update_ey_x<<<grid_size, block_size>>>(Ca_ey_x, Cb_ey_x, p_ey_x, p_hz, Nx, Ny, Nz);

    //     update_ez_x<<<grid_size, block_size>>>(Ca_ez_x, Cb_ez_x, p_ez_x, p_hy, Nx, Ny, Nz);
    //     update_ez_y<<<grid_size, block_size>>>(Ca_ez_y, Cb_ez_y, p_ez_y, p_hx, Nx, Ny, Nz);

    //     // Wait for the kernel to complete execution
    //     cudaDeviceSynchronize();

    //     combine_ex<<<grid_size, block_size>>>(p_ex, p_ex_y, p_ex_z, Nx, Ny, Nz);
    //     combine_ey<<<grid_size, block_size>>>(p_ey, p_ey_z, p_ey_x, Nx, Ny, Nz);
    //     combine_ez<<<grid_size, block_size>>>(p_ez, p_ez_x, p_ez_y, Nx, Ny, Nz);

    //     cudaDeviceSynchronize();

    //     // update h-fields
    //     update_hx_y<<<grid_size, block_size>>>(Da_hx_y, Db_hx_y1, Db_hx_y2, p_hx_y, p_ez, Nx, Ny, Nz);
    //     update_hx_z<<<grid_size, block_size>>>(Da_hx_z, Db_hx_z1, Db_hx_z2, p_hx_z, p_ey, Nx, Ny, Nz);

    //     update_hy_z<<<grid_size, block_size>>>(Da_hy_z, Db_hy_z1, Db_hy_z2, p_hy_z, p_ex, Nx, Ny, Nz);
    //     update_hy_x<<<grid_size, block_size>>>(Da_hy_x, Db_hy_x1, Db_hy_x2, p_hy_x, p_ez, Nx, Ny, Nz);

    //     update_hz_x<<<grid_size, block_size>>>(Da_hz_x, Db_hz_x1, Db_hz_x2, p_hz_x, p_ey, Nx, Ny, Nz);
    //     update_hz_y<<<grid_size, block_size>>>(Da_hz_y, Db_hz_y1, Db_hz_y2, p_hz_y, p_ex, Nx, Ny, Nz);

    //     cudaDeviceSynchronize();

    //     combine_hx<<<grid_size, block_size>>>(p_hx, p_hx_y, p_hx_z, Nx, Ny, Nz);
    //     combine_hy<<<grid_size, block_size>>>(p_hy, p_hy_z, p_hy_x, Nx, Ny, Nz);
    //     combine_hz<<<grid_size, block_size>>>(p_hz, p_hz_x, p_hz_y, Nx, Ny, Nz);

    //     cudaDeviceSynchronize();


    //     for (int i = 0; i < n_probes; i++)
    //     {   
    //         Probe * p = &(probes[i]);

    //         if (p->is_source)
    //         {
    //             update_source<<<1, 1>>>(p->field_p, p->field_s1_p, p->field_s2_p, (p->values)[n]);
    //         }

    //         // update probes
    //         // if this is a source, the values array is replaced with the resulting total voltage after it is used for
    //         // each time step.
    //         cudaMemcpy(((p->values) + n), (p->field_p), 4, cudaMemcpyDefault);
    //     }

    //     cudaDeviceSynchronize();
    // }
    
    // Clean up field arrays
    cudaFree(p_ex_y);
    cudaFree(p_ex_z);
    cudaFree(p_ex);

    cudaFree(p_ey_z);
    cudaFree(p_ey_x);
    cudaFree(p_ey);

    cudaFree(p_ez_x);
    cudaFree(p_ez_y);
    cudaFree(p_ez);

    cudaFree(p_hx_y);
    cudaFree(p_hx_z);
    cudaFree(p_hx);

    cudaFree(p_hy_z);
    cudaFree(p_hy_x);
    cudaFree(p_hy);

    cudaFree(p_hz_x);
    cudaFree(p_hz_y);
    cudaFree(p_hz);

    // clean up coefficients
    cudaFree(Ca_ex_y);
    cudaFree(Ca_ex_z);
    cudaFree(Cb_ex_y);
    cudaFree(Cb_ex_z);

    cudaFree(Ca_ey_z);
    cudaFree(Ca_ey_x);
    cudaFree(Cb_ey_z);
    cudaFree(Cb_ey_z);

    cudaFree(Ca_ez_x);
    cudaFree(Ca_ez_y);
    cudaFree(Cb_ez_x);
    cudaFree(Cb_ez_y);

    // hx coefficients
    cudaFree(Db_hx_y1);
    cudaFree(Db_hx_y2);
    cudaFree(Db_hx_z1);
    cudaFree(Db_hx_z2);

    cudaFree(Da_hx_y);
    cudaFree(Da_hx_z);

    // hy coefficients
    cudaFree(Db_hy_z1);
    cudaFree(Db_hy_z2);
    cudaFree(Db_hy_x1);
    cudaFree(Db_hy_x2);

    cudaFree(Da_hy_z);
    cudaFree(Da_hy_x);

    // hz coefficients
    cudaFree(Db_hz_x1);
    cudaFree(Db_hz_x2);
    cudaFree(Db_hz_y1);
    cudaFree(Db_hz_y2);

    cudaFree(Da_hz_x);
    cudaFree(Da_hz_y);

    // clean up probes
    cudaFree(probe_idx_dev);
    cudaFree(probe_type_dev);
    cudaFree(probe_values_dev);
    cudaFree(probe_is_src_dev);

}
