#pragma once

#include <cuda_runtime.h>
#include <string>
#include <cstdio>
#include "types.cuh"
#include "constant.cuh"

__host__ inline void cuda_call(cudaError_t err_code, const int line, const char* file) {
    if(err_code != cudaSuccess) {
        fprintf(stderr, "\nfile %s, line %d\nCUDA Error: %s (err_num=%d)\n", file, line, cudaGetErrorString(err_code), err_code);
        cudaDeviceReset();
        exit(1);
    }
}

__host__ inline void cuda_error_check(const char* kernel_name, const int line, const char* file) {
    if(cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "\nfile %s, line %d\nCUDA Kernel Function Error: %s %s\n", file, line, kernel_name, cudaGetErrorString(cudaGetLastError()));
        cudaDeviceReset();
        exit(1);
    }
}


inline __host__ bool fast_check_obstacle_cpu(const u32* obstacles, const u32 pitch, const u32 y, const u32 x) {
    return obstacles[y * pitch + (x >> 5)] & (1U << (x & 31U));
}

#define CUDA_CALL(x) cuda_call(x, __LINE__, __FILE__)

#define CUDA_ERROR_CHECK(kernel_name) cuda_error_check(kernel_name, __LINE__, __FILE__)

__host__ void die(const std::string& message, const int line, const char* file); 

__host__ void load_params(t_param& params, const char* path);

__host__ void print_cpu_speeds(const t_param& params, const t_cpu_data& cpu_data);

__host__ void init_cpu_data(const t_param& params, t_cpu_data& cpu_data);

__host__ void deinit_cpu_data(const t_param& params, t_cpu_data& cpu_data);

__host__ void init_gpu_data(const t_param& params, t_gpu_data& gpu_data, const char* obstacle_file);

__host__ void deinit_gpu_data(const t_param& params, t_gpu_data& gpu_data);

__host__ void speeds_host_to_device_async(const t_param& params, const t_cpu_data& cpu_data, const t_gpu_data& gpu_data, cudaStream_t cuda_stream = 0);

__host__ void speeds_device_to_host_async(const t_param& params, const t_cpu_data& cpu_data, const t_gpu_data& gpu_data, cudaStream_t cuda_stream = 0);

__device__ inline float load_3d_elem(const void* ptr, const u32 x, const u32 y, const u32 z, const u32 pitch, const u32 ysize) {
    return reinterpret_cast<const float*>(reinterpret_cast<const char*>(ptr) + pitch * (z * ysize + y))[x];
}

__device__ inline void save_3d_elem(void* ptr, const float val, const u32 x, const u32 y, const u32 z, const u32 pitch, const u32 ysize) {
    reinterpret_cast<float*>(reinterpret_cast<char*>(ptr) + pitch * (z * ysize + y))[x] = val;
}

__host__ inline void test_gpu(void) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties_v2(&prop, 0));
    printf("shared memory per block %llu\n", prop.sharedMemPerBlock);
}