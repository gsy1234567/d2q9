#pragma once

#include <cstdint>

using u32 = std::uint32_t;


struct t_param {
    u32    nx;            /* no. of cells in x-direction */
    u32    ny;            /* no. of cells in y-direction */
    u32    maxIters;      /* no. of iterations */
    float  density;       /* density per cell */
    float  viscosity;     /* kinematic viscosity of fluid */
    float  velocity;      /* inlet velocity */
    u32    type;          /* inlet type */
    float  omega;         /* relaxation parameter */
};

struct t_gpu_data {
    cudaPitchedPtr p_speeds;
    cudaExtent dev_speeds_extent;
    u32 obs_pitch;
    float *p_dev_inlets = nullptr;   
};

struct t_cpu_data {
    float *p_host_speeds = nullptr;
};

