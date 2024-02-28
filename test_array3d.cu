

int main() {
    
    // t_param params;
    // params.density = 1;
    // params.nx = 7;
    // params.ny = 2;
    // array3d<float, Host> array_h {params.nx, params.ny, 9};
    // array3d<float, Device> array_d {array_h};
    // float * host_speeds = (float*)malloc(sizeof(float) * params.nx * params.ny * 9);
    // cudaStream_t stream;

    // CUDA_CALL(cudaStreamCreate(&stream));
    // init_device_speeds(params, array_d, stream);
    // copy_device_speeds_to_host(params, array_d, host_speeds, stream);
    // CUDA_CALL(cudaStreamSynchronize(stream));
    // print_host_speeds(params, host_speeds);
    // CUDA_CALL(cudaStreamDestroy(stream));
    // free(host_speeds);
    CUDA_CALL(cudaFuncGetAttributes(&attr, &_d2q9_bgk));
    return 0;
}