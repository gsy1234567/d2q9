cmake_minimum_required(VERSION 3.23)

project(d2q9 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

file(
    GLOB 
    cpu_version_src 
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/src/*.cpp
)

add_executable(main_cpu main_cpu.cpp ${cpu_version_src})

target_include_directories(main_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/include/)

file(
    GLOB 
    gpu_version_src 
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_version/src/*.cu
)

add_executable(main_gpu main_gpu.cu ${gpu_version_src})

target_include_directories(main_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpu_version/include/)

