#include "array3d.cuh"
#include "types.cuh"
#include <iostream>
#include <cstdlib>
#include <cooperative_groups.h>
#include <fstream>


using namespace cooperative_groups;
using namespace gsy_cuda;

__constant__ u32 dev_obstacles [0x4000];
u32 host_obstacles[0x4000];

struct cpu_data_t {
    float *speeds;
};

struct gpu_data_t {
    array3d<float, Device> dev_speed_src;
    array3d<float, Device> dev_speed_dst;
    const u32 * constant_obsatcle;
    float * dev_inlets;
    u32 obstacles_pitch;
};

__device__ bool check_obstacle(u32 pitch, u32 x_idx, u32 y_idx) {
    return dev_obstacles[y_idx * pitch + (x_idx >> 5)] & (1U << (x_idx & 31U));
}

__host__ void set_obstacle(u32 pitch, u32 x_idx, u32 y_idx) {
    host_obstacles[y_idx * pitch + (x_idx >> 5)] |= (1U << (x_idx & 31U));
}

__device__ void load_speeds(u32 x_idx, u32 y_idx, float* local_tmp, array3d<float, Device> dev_speed_src) {
    local_tmp[0] = dev_speed_src.at(x_idx, y_idx, 0);
    local_tmp[1] = dev_speed_src.at(x_idx, y_idx, 1);
    local_tmp[2] = dev_speed_src.at(x_idx, y_idx, 2);
    local_tmp[3] = dev_speed_src.at(x_idx, y_idx, 3);
    local_tmp[4] = dev_speed_src.at(x_idx, y_idx, 4);
    local_tmp[5] = dev_speed_src.at(x_idx, y_idx, 5);
    local_tmp[6] = dev_speed_src.at(x_idx, y_idx, 6);
    local_tmp[7] = dev_speed_src.at(x_idx, y_idx, 7);
    local_tmp[8] = dev_speed_src.at(x_idx, y_idx, 8);
}

__device__ void collision(float* tmp, float omega) {

    constexpr float inv_c_sq = 3.f;
    constexpr float w0 = 4.f / 9.f;
    constexpr float w1 = 1.f / 9.f;
    constexpr float w2 = 1.f / 36.f;

    const float local_density = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] + tmp[8];
    const float u_x = (tmp[1] + tmp[5] + tmp[8] - tmp[3] - tmp[6] - tmp[7]) / local_density;
    const float u_y = (tmp[2] + tmp[5] + tmp[6] - tmp[4] - tmp[7] - tmp[8]) / local_density;
    const float u_sq = u_x * u_x + u_y * u_y;
    const float tmp_term_0 = u_sq * (0.5f * inv_c_sq);
    //handle speed 0~2
    float ori_tmp_0 = tmp[0] * (1.f - omega);
    float ori_tmp_1 = tmp[1] * (1.f - omega);
    float ori_tmp_2 = tmp[2] * (1.f - omega);
    tmp[0] = 0.f * inv_c_sq;
    tmp[1] = u_x * inv_c_sq;
    tmp[2] = u_y * inv_c_sq;
    tmp[0] = w0 * local_density * (1.f + tmp[0] * (1.f + 0.5f * tmp[0]) - tmp_term_0);
    tmp[1] = w1 * local_density * (1.f + tmp[1] * (1.f + 0.5f * tmp[1]) - tmp_term_0);
    tmp[2] = w1 * local_density * (1.f + tmp[2] * (1.f + 0.5f * tmp[2]) - tmp_term_0);
    tmp[0] = omega * tmp[0] + ori_tmp_0;
    tmp[1] = omega * tmp[1] + ori_tmp_1;
    tmp[2] = omega * tmp[2] + ori_tmp_2;
    //handle speed 3~5
    ori_tmp_0 = tmp[3] * (1.f - omega);
    ori_tmp_1 = tmp[4] * (1.f - omega);
    ori_tmp_2 = tmp[5] * (1.f - omega);
    tmp[3] = u_x * -inv_c_sq;
    tmp[4] = u_y * -inv_c_sq;
    tmp[5] = (u_x + u_y) * inv_c_sq;
    tmp[3] = w1 * local_density * (1.f + tmp[3] * (1.f + 0.5f * tmp[3]) - tmp_term_0);
    tmp[4] = w1 * local_density * (1.f + tmp[4] * (1.f + 0.5f * tmp[4]) - tmp_term_0);
    tmp[5] = w2 * local_density * (1.f + tmp[5] * (1.f + 0.5f * tmp[5]) - tmp_term_0);
    tmp[3] = omega * tmp[3] + ori_tmp_0;
    tmp[4] = omega * tmp[4] + ori_tmp_1;
    tmp[5] = omega * tmp[5] + ori_tmp_2;
    //handle speed 6~8
    ori_tmp_0 = tmp[6] * (1.f - omega);
    ori_tmp_1 = tmp[7] * (1.f - omega);
    ori_tmp_2 = tmp[8] * (1.f - omega);
    tmp[6] = (-u_x + u_y) * inv_c_sq;
    tmp[7] = (u_x + u_y) * -inv_c_sq;
    tmp[8] = (u_x - u_y) * inv_c_sq;
    tmp[6] = w2 * local_density * (1.f + tmp[6] * (1.f + 0.5f * tmp[6]) - tmp_term_0);
    tmp[7] = w2 * local_density * (1.f + tmp[7] * (1.f + 0.5f * tmp[7]) - tmp_term_0);
    tmp[8] = w2 * local_density * (1.f + tmp[8] * (1.f + 0.5f * tmp[8]) - tmp_term_0);
    tmp[6] = omega * tmp[6] + ori_tmp_0;
    tmp[7] = omega * tmp[7] + ori_tmp_1;
    tmp[8] = omega * tmp[8] + ori_tmp_2;

}

__device__ void obstacle(float* tmp) {
    float tmp0 = tmp[1];
    float tmp1 = tmp[2];
    float tmp2 = tmp[5];
    float tmp3 = tmp[6];
    tmp[1] = tmp[3];
    tmp[2] = tmp[4];
    tmp[5] = tmp[7];
    tmp[6] = tmp[8];
    tmp[3] = tmp0;
    tmp[4] = tmp1;
    tmp[7] = tmp2;
    tmp[8] = tmp3;
} 

__device__ void stream(float* tmp, const u32 x_idx, const u32 y_idx, const u32 nx, const u32 ny, array3d<float, Device> dev_speeds_dst) {
    dev_speeds_dst.at(x_idx, y_idx, 0) = tmp[0];

    if(x_idx + 1 < nx) {
        dev_speeds_dst.at(x_idx + 1, y_idx, 1) = tmp[1];
        if(y_idx + 1 < ny) {
            dev_speeds_dst.at(x_idx + 1, y_idx + 1, 5) = tmp[5];
        }
    }

    if(y_idx + 1 < ny) {
        dev_speeds_dst.at(x_idx, y_idx + 1, 2) = tmp[2];
        if(x_idx > 0) {
            dev_speeds_dst.at(x_idx - 1, y_idx + 1, 6) = tmp[6];
        }
    }

    if(x_idx > 0) {
        dev_speeds_dst.at(x_idx - 1, y_idx, 3) = tmp[3];
        if(y_idx > 0) {
            dev_speeds_dst.at(x_idx - 1, y_idx - 1, 7) = tmp[7];
        }
    }

    if(y_idx > 0) {
        dev_speeds_dst.at(x_idx, y_idx - 1, 4) = tmp[4];
        if(x_idx + 1 < nx) {
            dev_speeds_dst.at(x_idx + 1, y_idx - 1, 8) = tmp[8];
        }
    }
}

__device__ void boundary(float* tmp, const u32 x_idx, const u32 y_idx, const u32 nx, const u32 ny, array3d<float, Device> dev_speeds_dst, const float* dev_inlets) {

    //top wall
    if(y_idx == ny - 1) {
        dev_speeds_dst.at(x_idx, y_idx, 4) = tmp[2];
        dev_speeds_dst.at(x_idx, y_idx, 7) = tmp[5];
        dev_speeds_dst.at(x_idx, y_idx, 8) = tmp[6];
    }
    //bottom wall
    if(y_idx == 0) {
        dev_speeds_dst.at(x_idx, y_idx, 2) = tmp[4];
        dev_speeds_dst.at(x_idx, y_idx, 5) = tmp[7];
        dev_speeds_dst.at(x_idx, y_idx, 6) = tmp[8];
    }

    constexpr float cst1 = 2.f / 3.f;
    constexpr float cst2 = 1.f / 6.f;
    constexpr float cst3 = 1.f / 2.f;

    //left wall
    if(x_idx == 0) {
        tmp[0] = dev_speeds_dst.at(0, y_idx, 0);
        tmp[2] = dev_speeds_dst.at(0, y_idx, 2);
        tmp[3] = dev_speeds_dst.at(0, y_idx, 3);
        tmp[4] = dev_speeds_dst.at(0, y_idx, 4);
        tmp[6] = dev_speeds_dst.at(0, y_idx, 6);
        tmp[7] = dev_speeds_dst.at(0, y_idx, 7);
        const float inlets = dev_inlets[y_idx];
        const float local_density = ((tmp[0] + tmp[2] + tmp[4]) + 2.f * (tmp[3] + tmp[6] + tmp[7])) / (1.f - inlets);
        dev_speeds_dst.at(0, y_idx, 1) = tmp[3] + cst1 * local_density * inlets;
        dev_speeds_dst.at(0, y_idx, 5) = tmp[7] - cst3 * (tmp[2] - tmp[4]) + cst2 * local_density * inlets;
        dev_speeds_dst.at(0, y_idx, 8) = tmp[6] + cst3 * (tmp[2] - tmp[4]) + cst2 * local_density * inlets;
    }

    //right wall
    if(x_idx == nx - 1) {
        dev_speeds_dst.at(x_idx, y_idx, 0) = dev_speeds_dst.at(x_idx - 1, y_idx, 0);
        dev_speeds_dst.at(x_idx, y_idx, 1) = dev_speeds_dst.at(x_idx - 1, y_idx, 1);
        dev_speeds_dst.at(x_idx, y_idx, 2) = dev_speeds_dst.at(x_idx - 1, y_idx, 2);
        dev_speeds_dst.at(x_idx, y_idx, 3) = dev_speeds_dst.at(x_idx - 1, y_idx, 3);
        dev_speeds_dst.at(x_idx, y_idx, 4) = dev_speeds_dst.at(x_idx - 1, y_idx, 4);
        dev_speeds_dst.at(x_idx, y_idx, 5) = dev_speeds_dst.at(x_idx - 1, y_idx, 5);
        dev_speeds_dst.at(x_idx, y_idx, 6) = dev_speeds_dst.at(x_idx - 1, y_idx, 6);
        dev_speeds_dst.at(x_idx, y_idx, 7) = dev_speeds_dst.at(x_idx - 1, y_idx, 7);
        dev_speeds_dst.at(x_idx, y_idx, 8) = dev_speeds_dst.at(x_idx - 1, y_idx, 8);
    }
}

__global__ void _d2q9_bgk(t_param params, gpu_data_t gpu_data) {
    float tmp[9]; //locate in on-chip memory
    
    const u32 x_idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 y_idx    = blockIdx.y * blockDim.y + threadIdx.y;

    for(u32 i = 0 ; i < params.maxIters ; ++i) {
        if(x_idx < params.nx && y_idx < params.ny) {
            load_speeds(x_idx, y_idx, tmp, gpu_data.dev_speed_src);
            if(check_obstacle(gpu_data.obstacles_pitch, x_idx, y_idx)) {
                obstacle(tmp);
            } else {
                collision(tmp, params.omega);
            }
            stream(tmp, x_idx, y_idx, params.nx, params.ny, gpu_data.dev_speed_dst);
        }

        this_grid().sync();


        if(x_idx < params.nx && y_idx < params.ny) {
            boundary(tmp, x_idx, y_idx, params.nx, params.ny, gpu_data.dev_speed_dst, gpu_data.dev_inlets);
        }

        auto tmp = gpu_data.dev_speed_src._ptr.ptr;
        gpu_data.dev_speed_src._ptr.ptr = gpu_data.dev_speed_dst._ptr.ptr;
        gpu_data.dev_speed_dst._ptr.ptr =  tmp;

    }
}

__global__ void _init_device_speeds(array3d<float, Device> dev_speeds, float density, const u32 nx, const u32 ny) {
    const u32 x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const u32 y_idx = blockDim.y * blockIdx.y + threadIdx.y;
    if(x_idx < nx && y_idx < ny) {
        dev_speeds.at(x_idx, y_idx, 0) = density * 4.f / 9.f;
        dev_speeds.at(x_idx, y_idx, 1) = density / 9.f;
        dev_speeds.at(x_idx, y_idx, 2) = density / 9.f;
        dev_speeds.at(x_idx, y_idx, 3) = density / 9.f;
        dev_speeds.at(x_idx, y_idx, 4) = density / 9.f;
        dev_speeds.at(x_idx, y_idx, 5) = density / 36.f;
        dev_speeds.at(x_idx, y_idx, 6) = density / 36.f;
        dev_speeds.at(x_idx, y_idx, 7) = density / 36.f;
        dev_speeds.at(x_idx, y_idx, 8) = density / 36.f;
    }
}

__global__ void _init_device_inlets(u32 ny, float velocity, bool type, float* dev_inlets) {
    const u32 x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(x_idx < ny) {
        dev_inlets[x_idx] = type ? velocity * 4.f * (1.f - (float)x_idx / (float)ny) * ((float)x_idx + 1.f) / (float)ny : velocity;
    }
}

__host__ void init_device_data(const t_param& params, gpu_data_t& gpu_data,  const char* obstaclesfile, cudaStream_t stream = 0) {
    constexpr u32 block_x = 32;
    constexpr u32 block_y = 32;
    gpu_data.obstacles_pitch = (params.nx - 1) / 32 + 1;
    gpu_data.constant_obsatcle = dev_obstacles;
    if(gpu_data.obstacles_pitch * params.ny > 0x4000) {
        die("don't have enough constant memory!", __LINE__, __FILE__);
    }
    dim3 grid_info {(params.nx - 1) / block_x + 1, (params.ny - 1) / block_y + 1, 1};
    dim3 block_info {block_x, block_y, 1};
    _init_device_speeds<<<grid_info, block_info, 0, stream>>>(gpu_data.dev_speed_src, params.density, params.nx, params.ny);
    _init_device_inlets<<<(params.nx - 1) / block_x + 1, block_x, 0, stream>>>(params.ny, params.velocity, params.type, gpu_data.dev_inlets);

    std::ifstream obstacle_file{obstaclesfile};
    std::string line;
    int nx, ny, block;

    if(!obstacle_file) {
        die(std::string("Could not open the obstacle file:") + obstaclesfile, __LINE__, __FILE__);
    }

    while(std::getline(obstacle_file, line)) {
        if(sscanf(line.data(), "%d %d %d", &nx, &ny, &block) != 3) {
            die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
        }

        ny = ny + params.ny / 2;

        if(nx < 0 || nx >= params.nx) die("obstacle x-coord out of range", __LINE__, __FILE__);
        if(ny < 0 || ny >= params.ny) die("obstacle y-coord out of range", __LINE__, __FILE__);
        if(block != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
        set_obstacle(gpu_data.obstacles_pitch, nx, ny);
    }

    CUDA_CALL(cudaMemcpyToSymbolAsync(dev_obstacles, host_obstacles, sizeof(dev_obstacles), 0, cudaMemcpyDefault, stream));
}

__host__ void copy_device_speeds_to_host(const t_param& params, array3d<float, Device> dev_speeds, float* host_speeds, cudaStream_t stream = 0) {
    cudaMemcpy3DParms cpy_param {0};
    cpy_param.srcPtr = dev_speeds.get();
    cpy_param.dstPtr = make_cudaPitchedPtr(host_speeds, sizeof(float) * params.nx, params.nx, params.ny);
    cpy_param.extent = cudaExtent{sizeof(float) * params.nx, params.ny, 9};
    cpy_param.kind = cudaMemcpyDeviceToHost; 
    CUDA_CALL(cudaMemcpy3DAsync(&cpy_param, stream));
}

__host__ void print_host_speeds(const t_param& params, float* host_speeds) {
    for(int i = 0 ; i < 9 ; ++i) {
        printf("speeds: %d\n", i);
        for(int yy = params.ny - 1 ; yy >= 0 ; --yy) {
            for(int xx = 0 ; xx < params.nx ; ++xx) {
                printf("%.5f ", host_speeds[xx + (yy + params.ny * i) * params.nx]);
            }
            printf("\n");
        }
    }
}



inline void usage(const char* exe) {
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile> <output_directory>\n", exe);
  exit(1);
}

int main(int argc, char *argv[]) {
    char*    paramfile = nullptr;    /* name of the input parameter file */
    char*    obstaclefile = nullptr; /* name of a the input obstacle file */
    char*    out_dir = nullptr;      /* name of output directory */
    t_param  params;                 /* struct to hold parameter values */
    gpu_data_t gpu_data;
    cudaStream_t cuda_stream;
    CUDA_CALL(cudaStreamCreate(&cuda_stream));

    //handle input
    if(argc != 4) {
        usage(argv[0]);
    }

    paramfile = argv[1];
    obstaclefile = argv[2];
    out_dir = argv[3];
    //handle input end
    

    load_params(params, paramfile);
    array3d<float, Host> p_host_1 {params.nx, params.ny, 9};
    array3d<float, Host> p_host_2 {params.nx, params.ny, 9};
    float *host_speeds;
    CUDA_CALL(cudaMallocHost(&host_speeds, sizeof(float) * 9 * params.nx * params.ny));

    gpu_data.dev_speed_src = p_host_1;
    gpu_data.dev_speed_dst = p_host_2;
    CUDA_CALL(cudaMallocAsync(&gpu_data.dev_inlets, params.ny * sizeof(float), cuda_stream));
    init_device_data(params, gpu_data, obstaclefile, cuda_stream);
    dim3 grid_info {
        (params.nx - 1) / 32 + 1, 
        (params.ny - 1) / 8 + 1, 
        1
    };
    dim3 block_info {
        32, 8, 1
    };
    void * args[] = {&params, &gpu_data};
    CUDA_CALL(cudaLaunchCooperativeKernel((void*)_d2q9_bgk, grid_info, block_info, args, 0, cuda_stream));
    cuda_error_check("_d2q9_bgk", __LINE__, __FILE__);
    if(params.maxIters % 2 == 0) {
        copy_device_speeds_to_host(params, p_host_1, host_speeds, cuda_stream);
    } else {
        copy_device_speeds_to_host(params, p_host_2, host_speeds, cuda_stream);
    }

    CUDA_CALL(cudaStreamSynchronize(cuda_stream));
    print_host_speeds(params, host_speeds);
    CUDA_CALL(cudaFreeHost(host_speeds));
    CUDA_CALL(cudaFree(gpu_data.dev_inlets));
    return 0;
}