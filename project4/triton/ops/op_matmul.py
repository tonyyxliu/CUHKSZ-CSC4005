import torch

import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # TODO
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pass


def matmul(a, b, activation=""):
    # TODO
    matmul_kernel[param](
    )
    return c

if __name__ == "__main__":
    a = torch.randn(64, 728, device='cuda', dtype=torch.float32)
    b = torch.randn(728, 500, device='cuda', dtype=torch.float32)

    c = matmul(a, b)
    c_torch = torch.matmul(a, b)
    # print(c)
    # print(c_torch)
    print(torch.allclose(c, c_torch, atol=1e-3))