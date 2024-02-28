#pragma once

#include <cuda_runtime.h>
#include "thrust/device_ptr.h"
#include "utils.cuh"


namespace gsy_cuda {
    template<typename T, typename Location>
    class array3d;

    struct Host {};
    struct Device{};

    template<typename T>
    class array3d<T, Host> {
        public:
            __host__ __device__ array3d() : _ptr{0} {}

            __host__ array3d(u32 width, u32 height, u32 depth) {
                cudaExtent extent {width * sizeof(T), height, depth};
                CUDA_CALL(cudaMalloc3D(&_ptr, extent));
            }

            __host__ inline void swap(array3d& r) {
                std::swap(_ptr, r._ptr);
            }

            __host__ ~array3d() {
                CUDA_CALL(cudaFree(_ptr.ptr));
            }


            friend array3d<T, Device>;

            cudaPitchedPtr _ptr;
    };

    template<typename T>
    class array3d<T, Device> {
        public:
            struct index {
                u32 x;
                u32 y;
                u32 z;
            };
            __host__ __device__ array3d() : _ptr{0} {}
            __host__ array3d(const array3d<T, Host>& host_ptr) : _ptr{host_ptr._ptr} {}
            __host__ __device__ array3d(const array3d& other) : _ptr{other._ptr} {}
            __device__ T& at(u32 x, u32 y, u32 z) {
                return reinterpret_cast<T*>(reinterpret_cast<char*>(_ptr.ptr) + _ptr.pitch * (_ptr.ysize * z + y))[x];
            }
            __host__ __device__ void inline swap(array3d& r) {
                std::swap(_ptr, r._ptr);
            }
            __host__ cudaPitchedPtr get(void) const { return _ptr; }
            __device__ T& operator[](index index) {
                return reinterpret_cast<T*>(reinterpret_cast<char*>(_ptr.ptr) + _ptr.pitch * (_ptr.ysize * index.z + index.y))[index.x];
            }

            cudaPitchedPtr _ptr;
    };
}
