#pragma once

#include "types.cuh"

constexpr u32 thread_width = 32;
constexpr u32 max_cells    = 64 * 1024 * 8;
constexpr u32 max_obstacle_len = max_cells / 32;

