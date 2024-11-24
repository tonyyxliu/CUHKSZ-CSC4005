import torch
import triton
import triton.language as tl

# Triton kernel
@triton.jit
def add_bias_kernel(
    # TODO
):
    pass

# Python function
def add_bias(matrix, bias):
    # TODO
    add_bias_kernel[param](
        # TODO
    )
    return out

if __name__ == "__main__":

    matrix = torch.randn(128, 64, device='cuda', dtype=torch.float32)
    bias = torch.ones(64, device='cuda', dtype=torch.float32)

    out = add_bias(matrix, bias)
    out_ref = matrix + bias

    print(torch.allclose(out, out_ref))
