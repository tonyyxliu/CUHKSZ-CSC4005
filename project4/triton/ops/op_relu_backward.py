import torch
import triton
import triton.language as tl

# Triton kernel 实现：ReLU 反向传播
@triton.jit
def relu_backward_kernel(
    # TODO
):
    # TODO
    pass

# Python 函数，用于调用 Triton Kernel
def relu_backward(grad_output, input):
    # TODO

    # 调用 Triton kernel
    relu_backward_kernel[param](
    )

    return grad_input

if __name__ == "__main__":
    grad_output = torch.randn(128, 64, device='cuda', dtype=torch.float32)
    input = torch.randn(128, 64, device='cuda', dtype=torch.float32)

    grad_input = relu_backward(grad_output, input)
    grad_input_ref = grad_output.clone()
    grad_input_ref[input < 0] = 0
    print(torch.allclose(grad_input, grad_input_ref))