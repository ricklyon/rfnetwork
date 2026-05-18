#include <iostream>

#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

#include "solver.h"

// Ex update kernel
__global__ void update_ex_y(float* Ca_ex_y , float* ex_y, float* hz, int Nx, int Ny, int Nz)
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
    int hz0_idx = x_stride + ((y_idx) * Nz) + z_idx;
    int hz1_idx = x_stride + ((y_idx + 1) * Nz) + z_idx;

    ex_y[ex_idx] = Ca_ex_y[ex_idx] * ex_y[ex_idx] + (hz[hz1_idx] - hz[hz0_idx]);
}

__global__ void update_ex_z(float* Ca_ex_z , float* ex_z, float* hy, int Nx, int Ny, int Nz)
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
    int hy0_idx = x_stride + (y_idx * Nz) + z_idx;
    int hy1_idx = x_stride + (y_idx * Nz) + z_idx + 1;

    ex_z[ex_idx] = Ca_ex_z[ex_idx] * ex_z[ex_idx] + (hy[hy1_idx] - hy[hy0_idx]);
}

// Ey update kernel
__global__ void update_ey_z(float* Ca_ey_z , float* ey_z, float* hx, int Nx, int Ny, int Nz)
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
    int hx0_idx = x_stride + (y_idx * Nz) + z_idx;
    int hx1_idx = x_stride + (y_idx * Nz) + z_idx + 1;

    ey_z[ey_idx] = Ca_ey_z[ey_idx] * ey_z[ey_idx] + (hx[hx1_idx] - hx[hx0_idx]);
}

__global__ void update_ey_x(float* Ca_ey_x , float* ey_x, float* hz, int Nx, int Ny, int Nz)
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
    int hz0_idx = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hz1_idx = ((x_idx + 1) * Ny * Nz) + (y_idx * Nz) + z_idx;

    ey_x[ey_idx] = Ca_ey_x[ey_idx] * ey_x[ey_idx] + (hz[hz1_idx] - hz[hz0_idx]);
}

// Ez update kernel
__global__ void update_ez_x(float* Ca_ez_x , float* ez_x, float* hy, int Nx, int Ny, int Nz)
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
    int hy0_idx = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hy1_idx = ((x_idx + 1) * Ny * Nz) + (y_idx * Nz) + z_idx;

    ez_x[ez_idx] = Ca_ez_x[ez_idx] * ez_x[ez_idx] + (hy[hy1_idx] - hy[hy0_idx]);
}

__global__ void update_ez_y(float* Ca_ez_y , float* ez_y, float* hx, int Nx, int Ny, int Nz)
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
    int hx0_idx = (x_idx * Ny * Nz) + (y_idx * Nz) + z_idx;
    int hx1_idx = (x_idx * Ny * Nz) + ((y_idx + 1) * Nz) + z_idx;

    ez_y[ez_idx] = Ca_ez_y[ez_idx] * ez_y[ez_idx] + (hx[hx1_idx] - hx[hx0_idx]);
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

    // pointers to field component grids on devoce
    float * p_ex_y = nullptr;
    float * p_ex_z = nullptr;
    float * p_ex   = nullptr;

    float * p_ey_z = nullptr;
    float * p_ey_x = nullptr;
    float * p_ey   = nullptr;

    float * p_ez_x = nullptr;
    float * p_ez_y = nullptr;
    float * p_ez   = nullptr;

    float * p_hx_y = nullptr;
    float * p_hx_z = nullptr;
    float * p_hx   = nullptr;

    float * p_hy_z = nullptr;
    float * p_hy_x = nullptr;
    float * p_hy   = nullptr;

    float * p_hz_x = nullptr;
    float * p_hz_y = nullptr;
    float * p_hz   = nullptr;

    // pointers to coefficient arrays on device
    float * Ca_ex_y = nullptr;
    float * Ca_ex_z = nullptr;

    float * Ca_ey_z = nullptr;
    float * Ca_ey_x = nullptr;

    float * Ca_ez_x = nullptr;
    float * Ca_ez_y = nullptr;

    // allocate memory on device for fields
    cudaMalloc(&p_ex_y, f_size);
    cudaMalloc(&p_ex_z, f_size);
    cudaMalloc(&p_ex, f_size);

    cudaMalloc(&p_ey_z, f_size);
    cudaMalloc(&p_ey_x, f_size);
    cudaMalloc(&p_ey, f_size);

    cudaMalloc(&p_ez_x, f_size);
    cudaMalloc(&p_ez_y, f_size);
    cudaMalloc(&p_ez, f_size);

    cudaMalloc(&p_hx_y, f_size);
    cudaMalloc(&p_hx_z, f_size);
    cudaMalloc(&p_hx, f_size);

    cudaMalloc(&p_hy_z, f_size);
    cudaMalloc(&p_hy_x, f_size);
    cudaMalloc(&p_hy, f_size);

    cudaMalloc(&p_hz_x, f_size);
    cudaMalloc(&p_hz_y, f_size);
    cudaMalloc(&p_hz, f_size);

    // allocate memory on device for coefficients
    cudaMalloc(&Ca_ex_y, f_size);
    cudaMalloc(&Ca_ex_z, f_size);
    
    cudaMalloc(&Ca_ey_z, f_size);
    cudaMalloc(&Ca_ey_x, f_size);

    cudaMalloc(&Ca_ez_x, f_size);
    cudaMalloc(&Ca_ez_y, f_size);

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
    cudaMemcpy(Ca_ex_y, Cx.Ca_ex_y, f_size, cudaMemcpyDefault);
    cudaMemcpy(Ca_ex_z, Cx.Ca_ex_z, f_size, cudaMemcpyDefault);

    cudaMemcpy(Ca_ey_z, Cy.Ca_ey_z, f_size, cudaMemcpyDefault);
    cudaMemcpy(Ca_ey_x, Cy.Ca_ey_x, f_size, cudaMemcpyDefault);

    cudaMemcpy(Ca_ez_x, Cz.Ca_ez_x, f_size, cudaMemcpyDefault);
    cudaMemcpy(Ca_ez_y, Cz.Ca_ez_y, f_size, cudaMemcpyDefault);

    float * fields_base[6] = {p_ex, p_ey, p_ez, p_hx, p_hy, p_hz};
    float * fields_sp1_base[6] = {p_ex_y, p_ey_z, p_ez_x, p_hx_y, p_hy_z, p_hz_x};
    float * fields_sp2_base[6] = {p_ex_z, p_ey_x, p_ez_y, p_hx_z, p_hy_x, p_hz_y};

    dim3 block_size(Nx_th, Ny_th, Nz_th);
    dim3 grid_size(Nx_b, Ny_b, Nz_b);

    // main time stepping loop
    for (int n = 0; n < Nt; n++)
    {
        // update e-fields
        update_ex_y<<<grid_size, block_size>>>(Ca_ex_y, p_ex_y, p_hz, Nx, Ny, Nz);
        update_ex_z<<<grid_size, block_size>>>(Ca_ex_z, p_ex_z, p_hy, Nx, Ny, Nz);

        update_ey_z<<<grid_size, block_size>>>(Ca_ey_z, p_ex_y, p_hx, Nx, Ny, Nz);
        update_ey_x<<<grid_size, block_size>>>(Ca_ey_x, p_ex_y, p_hz, Nx, Ny, Nz);

        update_ez_x<<<grid_size, block_size>>>(Ca_ez_x, p_ez_x, p_hy, Nx, Ny, Nz);
        update_ez_y<<<grid_size, block_size>>>(Ca_ez_y, p_ez_y, p_hx, Nx, Ny, Nz);

        // Wait for the kernel to complete execution
        cudaDeviceSynchronize();

        combine_ex<<<grid_size, block_size>>>(p_ex, p_ex_y, p_ex_z, Nx, Ny, Nz);
        combine_ey<<<grid_size, block_size>>>(p_ey, p_ey_z, p_ey_x, Nx, Ny, Nz);
        combine_ez<<<grid_size, block_size>>>(p_ez, p_ez_x, p_ez_y, Nx, Ny, Nz);

        for (int i = 0; i < n_probes; i++)
        {   
            Probe * p = &(probes[i]);

            // get the pointer for the probe in the grid. 
            // TODO: offset to match the CUDA grid where all components have the same grid shape
            int p_offset = ((p->x_cell) * (p->NyNz)) + (p->yz_offset);
            float * p_field = fields_base[p->field_type] + p_offset;
            float * p_field_s1 = (fields_sp1_base[p->field_type]) + p_offset;
            float * p_field_s2 = (fields_sp2_base[p->field_type]) + p_offset;

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
    }



    // Clean Up
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


}
//unified-memory-end
