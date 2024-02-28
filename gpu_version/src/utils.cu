#include "utils.cuh"
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <bitset>
#include <iostream>

__host__ void die(const std::string& message, const int line, const char* file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message.data());
  fflush(stderr);
  exit(1);
}

__host__ void load_params(t_param& params, const char* path) {
    std::ifstream file {path};
    int match_cnt = 0;
    std::string line;

    if(!file) {
        die("Could not open the params file", __LINE__, __FILE__);
    }

    std::getline(file, line);
    match_cnt = sscanf(line.data(), "nx: %u", &params.nx);
    if(match_cnt != 1) die("Could not read param file: nx", __LINE__, __FILE__);

    std::getline(file, line);
    match_cnt = sscanf(line.data(), "ny: %u", &params.ny);
    if(match_cnt != 1) die("Could not read param file: ny", __LINE__, __FILE__);

    std::getline(file, line);
    match_cnt = sscanf(line.data(), "iters: %u", &params.maxIters);
    if(match_cnt != 1) die("Could not read param file: maxIters", __LINE__, __FILE__);

    std::getline(file, line);
    match_cnt = sscanf(line.data(), "density: %f", &params.density);
    if(match_cnt != 1) die("Could not read param file: density", __LINE__, __FILE__);

    std::getline(file, line);
    match_cnt = sscanf(line.data(), "viscosity: %f", &params.viscosity);
    if(match_cnt != 1) die("Could not read param file: viscosity", __LINE__, __FILE__);

    std::getline(file, line);
    match_cnt = sscanf(line.data(), "velocity: %f", &params.velocity);
    if(match_cnt != 1) die("Could not read param file: velocity", __LINE__, __FILE__);

    std::getline(file, line);
    match_cnt = sscanf(line.data(), "type: %u", &params.type);
    if(match_cnt != 1) die("Could not read param file: type", __LINE__, __FILE__);

    file.close();

    params.omega = 1.f / (3.f * params.viscosity + 0.5f);

    if(params.velocity > 0.2f) printf("Warning: There maybe computational instability due to compressibility.\n");

    if(2.f - params.omega < 0.15f) printf("Warning: Possible divergence of results due to relaxation time.\n");

    printf("==load paramter file==\n");
    printf("Param file: %s\n", path);
    printf("Number of cells:\t\t\t%u (%u x %u)\n",params.nx*params.ny,params.nx,params.ny);
    printf("Max iterations:\t\t\t\t%u\n", params.maxIters);
    printf("Density:\t\t\t\t%.6f\n", params.density);
    printf("Kinematic viscosity:\t\t\t%.6f\n", params.viscosity);
    printf("Inlet velocity:\t\t\t\t%.6f\n", params.velocity);
    printf("Inlet type:\t\t\t\t%u\n", params.type);
    printf("Relaxtion parameter:\t\t\t%.6f\n", params.omega);
}

__host__ void print_cpu_speeds(const t_param& params, const t_cpu_data& cpu_data) {
    for(u32 i = 0 ; i < 9 ; ++i) {
        printf("Speeds[%u]:\n", i);
        for(u32 yy = 0 ; yy < params.ny ; ++yy) {
            for(u32 xx = 0 ; xx < params.nx ; ++xx) {
                //std::ptrdiff_t offset = i * params.nx * params.ny + yy * params.nx + xx;
                std::ptrdiff_t offset = xx + params.nx * ( i * params.ny + yy);
                printf("%.5f ", cpu_data.p_host_speeds[offset]);
            }
            printf("\n");
        }
    }
}

__host__ void init_cpu_data(const t_param& params, t_cpu_data& cpu_data) {
    cpu_data.p_host_speeds = (float*)malloc(sizeof(float) * params.nx * params.ny * 9);
    auto p_start = cpu_data.p_host_speeds;
    const std::ptrdiff_t offset = params.nx * params.ny;
    std::fill_n(p_start, params.nx * params.ny, params.density * 4.f / 9.f);
    p_start += offset;

    
    for(int i = 0 ; i < 4 ; ++i) {
        std::fill_n(p_start, params.nx * params.ny, params.density / 9.f);
        p_start += offset;
    }

    for(int i = 0 ; i < 4 ; ++i) {
      std::fill_n(p_start, params.nx * params.ny, params.density / 36.f);
      p_start += offset;
    }
}

__host__ void deinit_cpu_data(const t_param& params, t_cpu_data& cpu_data) {
    free(cpu_data.p_host_speeds);
    cpu_data.p_host_speeds = nullptr;
}

// __host__ void init_gpu_data(const t_param& params, t_gpu_data& gpu_data, const char* obstacles_path) {
//     gpu_data.dev_speeds_extent = make_cudaExtent(sizeof(float) * params.nx, params.ny, 9);
//     CUDA_CALL(cudaMalloc3D(&gpu_data.p_speeds, gpu_data.dev_speeds_extent));

//     std::ifstream file {obstacles_path};
//     std::string line;

//     if(!file) {
//         die("Could not open the obstacles file", __LINE__, __FILE__);
//     }
  
//     const u32 u32_num = (params.nx + 31) / 32 * params.ny;
//     u32 cells_num = u32_num * 32;

//     if(cells_num > max_cells) {
//         die("Could not apply contant memory optimization", __LINE__, __FILE__);
//     }

//     gpu_data.obs_pitch = (params.nx + 31) / 32; 
//     // u32 * cpu_obstacles = (u32*)calloc(u32_num, sizeof(u32));
//     u32 cpu_obstacles[max_obstacle_len] = {0};

//     int ok = 0;
//     int x, y, block;

//     while(std::getline(file, line)) {
//         ok = sscanf(line.data(), "%d %d %d", &x, &y, &block);
//         y += params.ny / 2;
//         if(ok != 3) die("invalid format", __LINE__, __FILE__);
//         if(block != 1) die("obstacles file is not valid", __LINE__, __FILE__);
//         if(x >= params.nx) die("x is not valid", __LINE__, __FILE__);
//         if(y >= params.ny) die("y is not valid", __LINE__, __FILE__);
//         u32 x_offset = x / 32;
//         u32 bit_offset = x % 32;
//         cpu_obstacles[y * gpu_data.obs_pitch + x_offset] |= (1U << bit_offset);
//     }

//     std::cout << "==obstacles==" << std::endl;
//     for(u32 yy = 0 ; yy < params.ny ; ++yy) {
//         int x = 0;
//         for(u32 xx = 0 ; xx < params.nx ; xx += 32, ++x) {
//             std::bitset<32>& curr = reinterpret_cast<std::bitset<32>&>(cpu_obstacles[yy * gpu_data.obs_pitch + x]);
//             for(int i = 0 ; i < 32 ; ++i) {
//               std::cout << curr[i];
//             }
//         }
//         std::cout << std::endl;
//     }

//     CUDA_CALL(cudaMemcpyToSymbol(obstacles, cpu_obstacles, sizeof(cpu_obstacles)));
//     CUDA_CALL(cudaMalloc(&gpu_data.p_dev_inlets, sizeof(float) * params.ny));

//     float * cpu_inlets = (float*)malloc(sizeof(float) * params.ny);
//     if(!params.type) {
//       for(int jj = 0 ; jj < params.ny ; ++jj) {
//         cpu_inlets[jj] = params.velocity;
//       }
//     } else {
//       for(int jj = 0 ; jj < params.ny ; ++jj) {
//         cpu_inlets[jj] = params.velocity * 4.0f *((1.f-((float)jj)/params.ny)*((float)(jj+1))/params.ny);
//       }
//     }

//     CUDA_CALL(cudaMemcpy(gpu_data.p_dev_inlets, cpu_inlets, sizeof(float) * params.ny, cudaMemcpyDefault));
//     free(cpu_inlets);
// }

__host__ void deinit_gpu_data(const t_param& params, t_gpu_data& gpu_data) {
  CUDA_CALL(cudaFree(gpu_data.p_speeds.ptr));
  CUDA_CALL(cudaFree(gpu_data.p_dev_inlets));
}

__host__ void speeds_host_to_device_async(const t_param& params, const t_cpu_data& cpu_data, const t_gpu_data& gpu_data, cudaStream_t cuda_stream) {
    cudaMemcpy3DParms cpy_param {0};
    cpy_param.srcPtr = make_cudaPitchedPtr(cpu_data.p_host_speeds, sizeof(float) * params.nx, params.nx, params.ny);
    cpy_param.dstPtr = gpu_data.p_speeds;
    cpy_param.extent = gpu_data.dev_speeds_extent;
    cpy_param.kind   = cudaMemcpyHostToDevice;
    CUDA_CALL(cudaMemcpy3DAsync(&cpy_param, cuda_stream));
}

__host__ void speeds_device_to_host_async(const t_param& params, const t_cpu_data& cpu_data, const t_gpu_data& gpu_data, cudaStream_t cuda_stream ) {
    cudaMemcpy3DParms cpy_param{0};
    cpy_param.srcPtr = gpu_data.p_speeds;
    cpy_param.dstPtr = make_cudaPitchedPtr(cpu_data.p_host_speeds, sizeof(float) * params.nx, params.nx, params.ny);
    cpy_param.extent = gpu_data.dev_speeds_extent;
    cpy_param.kind   = cudaMemcpyDeviceToHost;
    CUDA_CALL(cudaMemcpy3DAsync(&cpy_param, cuda_stream));
}



