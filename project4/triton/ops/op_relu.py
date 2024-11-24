import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    # TODO
):
    # TODO
    pass

def relu(matrix):
    # TODO
    relu_kernel[param](
    )

    return out

if __name__ == "__main__":
    matrix = torch.randn(128, 64, device='cuda', dtype=torch.float32)

    out = relu(matrix)
    out_ref = torch.relu(matrix)
    print(torch.allclose(out, out_ref))