#include <cstdio>
#include "types.cuh"
#include "utils.cuh"


inline void usage(const char* exe) {
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile> <output_directory>\n", exe);
  exit(1);
}

int main(int argc, char *argv[]) {
    char*    paramfile = nullptr;    /* name of the input parameter file */
    char*    obstaclefile = nullptr; /* name of a the input obstacle file */
    char*    out_dir = nullptr;      /* name of output directory */
    t_param  params;                 /* struct to hold parameter values */
    t_cpu_data cpu_data;                
    t_gpu_data gpu_data;
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

    init_cpu_data(params, cpu_data);
    init_gpu_data(params, gpu_data, obstaclefile);
    //speeds_host_to_device_async(params, cpu_data, gpu_data);
    //speeds_device_to_host_async(params, cpu_data, gpu_data);
    print_cpu_speeds(params, cpu_data);
    deinit_gpu_data(params, gpu_data);
    deinit_cpu_data(params, cpu_data);
    CUDA_CALL(cudaStreamDestroy(cuda_stream));
    return 0;
}