import torch
import triton
import triton.language as tl

@triton.jit
def sum_kernel(
    # TODO
):
    pass
    # TODO


def sum_dim(matrix, dim):

    # TODO

    sum_kernel[param](
    )

    return out

if __name__ == "__main__":
    matrix = torch.randn(128, 64, device='cuda', dtype=torch.float32)
    out_dim0 = sum_dim(matrix, dim=0)
    out_dim1 = sum_dim(matrix, dim=1)

    out_ref_dim0 = torch.sum(matrix, dim=0)
    out_ref_dim1 = torch.sum(matrix, dim=1)
    print(torch.allclose(out_dim0, out_ref_dim0))
    print(torch.allclose(out_dim1, out_ref_dim1))