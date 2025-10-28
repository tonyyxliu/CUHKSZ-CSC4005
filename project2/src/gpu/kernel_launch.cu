#include <kernels.cuh>

void launch_kernel(char* kernel_name, MAT_DATATYPE* dMatA, MAT_DATATYPE* dMatB,
                   MAT_DATATYPE* dMatC, int M, int N, int K)
{
    // printf("%s\n", kernel_name);
    if (strcmp(kernel_name, "shared_mem_tiling") == 0)
    {
        printf("Running Kernel: shared_mem_tiling\n");
        launch_matmul_sm_tiling(dMatA, dMatB, dMatC, M, N, K);
    }
    // TODO: 速度不稳定，时快时慢
    if (strcmp(kernel_name, "matmul_1d_tiling") == 0)
    {
        printf("Running Kernel: matmul_1d_tiling\n");
        launch_matmul_1d_tiling(dMatA, dMatB, dMatC, N, M, K);
    }
}