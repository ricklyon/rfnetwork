#include <iostream>

#include <cuda_runtime_api.h>
#include <stdio.h>
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

struct EFieldCoefficients
{
    float* Ca_ex_y;
    float* Cb_ex_y;

    float* Ca_ex_z;
    float* Cb_ex_z;

    float* Ca_ey_z;
    float* Cb_ey_z; 

    float* Ca_ey_x;
    float* Cb_ey_x; 

    float* Ca_ez_x;
    float* Cb_ez_x; 

    float* Ca_ez_y;
    float* Cb_ez_y; 
};

struct HFieldCoefficients
{
    float* Da_hx_y;
    float* Db_hx_y1;
    float* Db_hx_y2;

    float* Da_hx_z;
    float* Db_hx_z1;
    float* Db_hx_z2;

    float* Da_hy_z;
    float* Db_hy_z1;
    float* Db_hy_z2;

    float* Da_hy_x;
    float* Db_hy_x1;
    float* Db_hy_x2;

    float* Da_hz_x;
    float* Db_hz_x1;
    float* Db_hz_x2;

    float* Da_hz_y;
    float* Db_hz_y1;
    float* Db_hz_y2;
};

struct Fields
{
    float* ex;
    float* ex_y;
    float* ex_z;

    float* ey;
    float* ey_z;
    float* ey_x;

    float* ez;
    float* ez_x;
    float* ez_y;

    float* hx;
    float* hx_y;
    float* hx_z;

    float* hy;
    float* hy_z;
    float* hy_x;

    float* hz;
    float* hz_x;
    float* hz_y;

};


__global__ void efield_update_kernel(
    EFieldCoefficients C, Fields F,
    int Nx, int Ny, int Nz, int Nt
)
{
    // cg::grid_group grid = cg::this_grid();

    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny)) | (z_idx >= (Nz)) )
    {
        return;
    }

    // field index, applies to e and h fields
    int f_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;


    // printf("idx = %d, %d, %d, fidx=%d\n", x_idx, y_idx, z_idx, f_idx);

    // h-field indices on either side of e-fields
    int hz_y2_idx = (x_idx * Ny * Nz) + ((y_idx + 1) * Nz) + z_idx;
    int hy_z2_idx = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx + 1;

    int hx_z2_idx = (x_idx * Ny * Nz)  + (y_idx * Nz) + z_idx + 1;
    int hz_x2_idx = ((x_idx + 1) * Ny * Nz) + (y_idx * Nz) + z_idx;

    int hy_x2_idx = ((x_idx + 1) * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hx_y2_idx = (x_idx * Ny * Nz) + ((y_idx + 1) * Nz) + z_idx;

    // main time stepping loop
    // for (int n = 0; n < Nt; n++)
    // {
    // update ex_y
    if (y_idx < (Ny - 1))
    {
        F.ex_y[f_idx] = C.Ca_ex_y[f_idx] * F.ex_y[f_idx] + C.Cb_ex_y[f_idx] * (F.hz[hz_y2_idx] - F.hz[f_idx]);
    }
    
    // update ex_z
    if (z_idx < (Nz - 1))
    {
        F.ex_z[f_idx] = C.Ca_ex_z[f_idx] * F.ex_z[f_idx] + C.Cb_ex_z[f_idx] * (F.hy[hy_z2_idx] - F.hy[f_idx]);
    }

    // update ey_z
    if (z_idx < (Nz - 1))
    {
        F.ey_z[f_idx] = C.Ca_ey_z[f_idx] * F.ey_z[f_idx] + C.Cb_ey_z[f_idx] * (F.hx[hx_z2_idx] - F.hx[f_idx]);
    }

    // update ey_x
    if (x_idx < (Nx - 1))
    {
        F.ey_x[f_idx] = C.Ca_ey_x[f_idx] * F.ey_x[f_idx] + C.Cb_ey_x[f_idx] * (F.hz[hz_x2_idx] - F.hz[f_idx]);
    }

    // update ez_x
    if (x_idx < (Nx - 1))
    {
        F.ez_x[f_idx] = C.Ca_ez_x[f_idx] * F.ez_x[f_idx] + C.Cb_ez_x[f_idx] * (F.hy[hy_x2_idx] - F.hy[f_idx]);
    }
    
    //update ez_y
    if (y_idx < (Ny - 1))
    {
        F.ez_y[f_idx] = C.Ca_ez_y[f_idx] * F.ez_y[f_idx] + C.Cb_ez_y[f_idx] * (F.hx[hx_y2_idx] - F.hx[f_idx]);
    }

    // combine E fields
    F.ex[f_idx] = F.ex_y[f_idx] + F.ex_z[f_idx];
    F.ey[f_idx] = F.ey_z[f_idx] + F.ey_x[f_idx];
    F.ez[f_idx] = F.ez_x[f_idx] + F.ez_y[f_idx];

    // grid.sync();
}

__global__ void hfield_update_kernel(
    HFieldCoefficients D, Fields F,
    int Nx, int Ny, int Nz, int Nt
)
{
    // cg::grid_group grid = cg::this_grid();
    
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int z_idx = threadIdx.z + blockIdx.z * blockDim.z;

    // skip update if thread is after the global grid boundary. 
    if ((x_idx >= (Nx)) | (y_idx >= (Ny)) | (z_idx >= (Nz)) )
    {
        return;
    }

    // field index, applies to e and h fields
    int f_idx  = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;

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


    // update hx_y
    if (y_idx > 0)
    {
        ez_y1 = F.ez[ez_y1_idx];
    }
    F.hx_y[f_idx] = D.Da_hx_y[f_idx] * F.hx_y[f_idx] + (D.Db_hx_y2[f_idx] * F.ez[f_idx]) - (D.Db_hx_y1[f_idx] * ez_y1);

    // update hx_z
    if (z_idx > 0)
    {
        ey_z1 = F.ey[ey_z1_idx];
    }
    F.hx_z[f_idx] = D.Da_hx_z[f_idx] * F.hx_z[f_idx] + (D.Db_hx_z2[f_idx] * F.ey[f_idx]) - (D.Db_hx_z1[f_idx] * ey_z1);

    // update hy_z
    if (z_idx > 0)
    {
        ex_z1 = F.ex[ex_z1_idx];
    }
    F.hy_z[f_idx] = D.Da_hy_z[f_idx] * F.hy_z[f_idx] + (D.Db_hy_z2[f_idx] * F.ex[f_idx]) - (D.Db_hy_z1[f_idx] * ex_z1);

    // update hy_x
    if (x_idx > 0)
    {
        ez_x1 = F.ez[ez_x1_idx];
    }
    F.hy_x[f_idx] = D.Da_hy_x[f_idx] * F.hy_x[f_idx] + (D.Db_hy_x2[f_idx] * F.ez[f_idx]) - (D.Db_hy_x1[f_idx] * ez_x1);

    // update hz_x
    if (x_idx > 0)
    {
        ey_x1 = F.ey[ey_x1_idx];
    }
    F.hz_x[f_idx] = D.Da_hz_x[f_idx] * F.hz_x[f_idx] + (D.Db_hz_x2[f_idx] * F.ey[f_idx]) - (D.Db_hz_x1[f_idx] * ey_x1);

    // update hz_y
    if (y_idx > 0)
    {
        ex_y1 = F.ex[ex_y1_idx];
    }
    F.hz_y[f_idx] = D.Da_hz_y[f_idx] * F.hz_y[f_idx] + (D.Db_hz_y2[f_idx] * F.ex[f_idx]) - (D.Db_hz_y1[f_idx] * ex_y1);

    // combine h-fields
    F.hx[f_idx] = F.hx_y[f_idx] + F.hx_z[f_idx];
    F.hy[f_idx] = F.hy_z[f_idx] + F.hy_x[f_idx];
    F.hz[f_idx] = F.hz_x[f_idx] + F.hz_y[f_idx];
}

__global__ void e_probe_update_kernel(
    Fields F, int n_probes, int n, int Nt, 
    int* probe_idx, int* probe_type, int* probe_is_src, float* probe_values
)
{

    int p_idx = threadIdx.x + blockIdx.x * blockDim.x;

    // skip update if thread is after the global grid boundary. 
    if ((p_idx >= (n_probes)))
    {
        return;
    }

    // field index, applies to e and h fields
    int f_idx  = probe_idx[p_idx];


    if (probe_type[p_idx] == 0)
    {
        if (probe_is_src[p_idx])
        {
            F.ex_y[f_idx] += probe_values[(p_idx * Nt) + n];
            F.ex_z[f_idx] += probe_values[(p_idx * Nt) + n];
            F.ex[f_idx] = F.ex_y[f_idx] + F.ex_z[f_idx];
        }
        probe_values[(p_idx * Nt) + n] = F.ex[f_idx];
    }
    else if (probe_type[p_idx] == 1)
    {
        if (probe_is_src[p_idx])
        {
            F.ey_z[f_idx] += probe_values[(p_idx * Nt) + n];
            F.ey_x[f_idx] += probe_values[(p_idx * Nt) + n];
            F.ey[f_idx] = F.ey_z[f_idx] + F.ey_x[f_idx];
        }
        probe_values[(p_idx * Nt) + n] = F.ey[f_idx];
    }
    else if (probe_type[p_idx] == 2)
    {
        if (probe_is_src[p_idx])
        {
            F.ez_x[f_idx] += probe_values[(p_idx * Nt) + n];
            F.ez_y[f_idx] += probe_values[(p_idx * Nt) + n];
            F.ez[f_idx] = F.ez_x[f_idx] + F.ez_y[f_idx];
        }
        probe_values[(p_idx * Nt) + n] = F.ez[f_idx];
    }

}

__global__ void h_probe_update_kernel(
    Fields F, int n_probes, int n, int Nt, 
    int* probe_idx, int* probe_type, int* probe_is_src, float* probe_values
)
{

    int p_idx = threadIdx.x + blockIdx.x * blockDim.x;

    // skip update if thread is after the global grid boundary. 
    if ((p_idx >= (n_probes)))
    {
        return;
    }

    // field index, applies to e and h fields
    int f_idx  = probe_idx[p_idx];


    if (probe_type[p_idx] == 3)
    {
        if (probe_is_src[p_idx])
        {
            F.hx_y[f_idx] += probe_values[(p_idx * Nt) + n];
            F.hx_z[f_idx] += probe_values[(p_idx * Nt) + n];
            F.hx[f_idx] = F.hx_y[f_idx] + F.hx_z[f_idx];
        }
        probe_values[(p_idx * Nt) + n] = F.hx[f_idx];
    }
    else if (probe_type[p_idx] == 4)
    {
        if (probe_is_src[p_idx])
        {
            F.hy_z[f_idx] += probe_values[(p_idx * Nt) + n];
            F.hy_x[f_idx] += probe_values[(p_idx * Nt) + n];
            F.hy[f_idx] = F.hy_z[f_idx] + F.hy_x[f_idx];
        }
        probe_values[(p_idx * Nt) + n] = F.hy[f_idx];
    }
    else if (probe_type[p_idx] == 5)
    {
        if (probe_is_src[p_idx])
        {
            F.hz_x[f_idx] += probe_values[(p_idx * Nt) + n];
            F.hz_y[f_idx] += probe_values[(p_idx * Nt) + n];
            F.hz[f_idx] = F.hz_x[f_idx] + F.hz_y[f_idx];
        }
        probe_values[(p_idx * Nt) + n] = F.hz[f_idx];
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
    int probe_is_src[MAX_PROBES];

    int * probe_idx_dev = nullptr;
    int * probe_type_dev = nullptr;
    float * probe_values_dev = nullptr;
    int * probe_is_src_dev = nullptr;

    cudaMalloc(&probe_idx_dev, n_probes * sizeof(int));
    cudaMalloc(&probe_type_dev, n_probes * sizeof(int));
    cudaMalloc(&probe_values_dev, n_probes * Nt * sizeof(float));
    cudaMalloc(&probe_is_src_dev, n_probes * sizeof(int));

    for (int i = 0; i < n_probes; i++)
    {   
        Probe * p = &(probes[i]);

        probe_idx[i] = ((p->x_cell) * Ny * Nz) + ((p->y_cell) * Nz) + ((p->z_cell));
        probe_type[i] = p->field_type;
        probe_is_src[i] = p->is_source;

        // printf("probe_idx = %d, %d\n", probe_idx[i], p->field_type);

        cudaMemcpy(probe_values_dev + (i * Nt), p->values, Nt * sizeof(float), cudaMemcpyDefault);
    }

    // size of thread blocks in each direction
    int Np_th = 8;

    // number of blocks in the grid for probes
    int Np_b = (n_probes + Np_th - 1) / Np_th;

    cudaMemcpy(probe_idx_dev, probe_idx, n_probes * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(probe_type_dev, probe_type, n_probes * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(probe_is_src_dev, probe_is_src, n_probes * sizeof(int), cudaMemcpyDefault);

    EFieldCoefficients ecoeff;
    
    ecoeff.Ca_ex_y = Ca_ex_y;
    ecoeff.Cb_ex_y = Cb_ex_y;

    ecoeff.Ca_ex_z = Ca_ex_z;
    ecoeff.Cb_ex_z = Cb_ex_z;

    ecoeff.Ca_ey_z = Ca_ey_z;
    ecoeff.Cb_ey_z = Cb_ey_z;

    ecoeff.Ca_ey_x = Ca_ey_x;
    ecoeff.Cb_ey_x = Cb_ey_x;

    ecoeff.Ca_ez_x = Ca_ez_x;
    ecoeff.Cb_ez_x = Cb_ez_x;

    ecoeff.Ca_ez_y = Ca_ez_y;
    ecoeff.Cb_ez_y = Cb_ez_y;

    HFieldCoefficients hcoeff;

    hcoeff.Da_hx_y  = Da_hx_y;
    hcoeff.Db_hx_y1 = Db_hx_y1;
    hcoeff.Db_hx_y2 = Db_hx_y2;

    hcoeff.Da_hx_z  = Da_hx_z;
    hcoeff.Db_hx_z1 = Db_hx_z1;
    hcoeff.Db_hx_z2 = Db_hx_z2;

    hcoeff.Da_hy_z  = Da_hy_z;
    hcoeff.Db_hy_z1 = Db_hy_z1;
    hcoeff.Db_hy_z2 = Db_hy_z2;

    hcoeff.Da_hy_x  = Da_hy_x;
    hcoeff.Db_hy_x1 = Db_hy_x1;
    hcoeff.Db_hy_x2 = Db_hy_x2;

    hcoeff.Da_hz_x  = Da_hz_x;
    hcoeff.Db_hz_x1 = Db_hz_x1;
    hcoeff.Db_hz_x2 = Db_hz_x2;

    hcoeff.Da_hz_y  = Da_hz_y;
    hcoeff.Db_hz_y1 = Db_hz_y1;
    hcoeff.Db_hz_y2 = Db_hz_y2;

    Fields fields;

    fields.ex = p_ex;
    fields.ex_y = p_ex_y;
    fields.ex_z = p_ex_z;

    fields.ey = p_ey;
    fields.ey_z = p_ey_z;
    fields.ey_x = p_ey_x;

    fields.ez = p_ez;
    fields.ez_x = p_ez_x;
    fields.ez_y = p_ez_y;

    fields.hx = p_hx;
    fields.hx_y = p_hx_y;
    fields.hx_z = p_hx_z;

    fields.hy = p_hy;
    fields.hy_z = p_hy_z;
    fields.hy_x = p_hy_x;

    fields.hz = p_hz;
    fields.hz_x = p_hz_x;
    fields.hz_y = p_hz_y;

    dim3 block_size(Nx_th, Ny_th, Nz_th);
    dim3 grid_size(Nx_b, Ny_b, Nz_b);

    printf("grid_size = %d\n", grid_size);
    printf("block_size = %d\n", block_size);

    cudaError_t err;

    for (int n = 0; n < Nt; n++)
    {
        efield_update_kernel<<<grid_size, block_size>>>(
            ecoeff, fields,
            Nx, Ny, Nz, Nt
        );

        e_probe_update_kernel<<<Np_b, Np_th>>>(
            fields, n_probes, n, Nt,
            probe_idx_dev, probe_type_dev, probe_is_src_dev, probe_values_dev
        );
        // err = cudaGetLastError();
        // printf("e-launch: %s\n", cudaGetErrorString(err));

        // cudaDeviceSynchronize(); 

        // err = cudaGetLastError();
        // printf("e-runtime: %s\n", cudaGetErrorString(err));

        hfield_update_kernel<<<grid_size, block_size>>>(
            hcoeff, fields,
            Nx, Ny, Nz, Nt
        );


        // err = cudaGetLastError();
        // printf("h-launch: %s\n", cudaGetErrorString(err));

        // cudaDeviceSynchronize(); 

        // err = cudaGetLastError();
        // printf("h-runtime: %s\n", cudaGetErrorString(err));

        h_probe_update_kernel<<<Np_b, Np_th>>>(
            fields, n_probes, n, Nt,
            probe_idx_dev, probe_type_dev, probe_is_src_dev, probe_values_dev
        );


        // err = cudaGetLastError();
        // printf("p-launch: %s\n", cudaGetErrorString(err));

        // cudaDeviceSynchronize(); 

        // err = cudaGetLastError();
        // printf("p-runtime: %s\n", cudaGetErrorString(err));
    }


    cudaDeviceSynchronize(); 

    err = cudaGetLastError();
    printf("ERROR: %s\n", cudaGetErrorString(err));

    printf("n_probes = %d\n", n_probes);
    // copy probe values from device to CPU
    for (int i = 0; i < n_probes; i++)
    {   
        Probe * p = &(probes[i]);

        cudaMemcpy(p->values, probe_values_dev + (i * Nt), Nt * sizeof(float), cudaMemcpyDefault);
    }
    
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
