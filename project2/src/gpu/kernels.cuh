#ifndef __KERNELS__
#define __KERNELS__

#pragma once
#include <matrix_lowPre.hpp>
#include "kernels/shared_mem_tiling.cuh"
#include "kernels/1D_tiling.cuh"

void launch_kernel(char* kernel_name, MAT_DATATYPE* dMatA, MAT_DATATYPE* dMatB,
                   MAT_DATATYPE* dMatC, int M, int N, int K);

#endif