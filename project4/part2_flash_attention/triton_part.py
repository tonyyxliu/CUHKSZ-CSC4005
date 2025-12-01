import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def flash_attention_v1(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, d_model: tl.constexpr,
    stride_qm, stride_km, stride_vm, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):  
    # Your code here
    # TODO

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def call_flash_attention_v1(q, k, v):
    assert q.shape == k.shape == v.shape, "Input shapes must match"
    assert q.dim() == 2, "Only support 2D input: (seq_len, d_model)"
    seq_len, d_model = q.shape
    o = torch.empty_like(q)

    # Your code here
    # TODO
    
    return o

def pytorch_attention(q, k, v):
        d_model = q.shape[-1]
        scale_factor = 1.0 / (d_model**0.5)
        scores = q @ k.T * scale_factor
        softmax = torch.softmax(scores, dim=1)
        attention = softmax @ v
        return attention

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[2 ** i for i in range(4, 13)],
        line_arg='provider',
        line_vals=['flash_attention', 'native'],
        line_names=["Flash Attention", "Native Attention"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="ms",
        plot_name="attention-performance",
        args={'d_model': 256},
    )
)
def benchmark(seq_len, d_model, provider):
    q = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    
    if provider == 'flash_attention':
        ms = triton.testing.do_bench(lambda: call_flash_attention_v1(q, k, v))
    if provider == 'native':
        ms = triton.testing.do_bench(lambda: pytorch_attention(q, k, v))
    
    return ms

def unit_test(seq_len, d_model):
    torch.manual_seed(0)
    q = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)

    o_triton = call_flash_attention_v1(q, k, v)
    o_torch = pytorch_attention(q, k, v)

    assert torch.allclose(o_triton, o_torch, atol=1e-3, rtol=1e-3), (o_triton, o_torch)
    print(f"Attention output correct for seq_len={seq_len}, d_model={d_model}!")

if __name__ == "__main__":
    for i in range(6, 12):
        unit_test(2 ** i, 64)
    print("pass all unit test")
    print("-----------------------------------------")    
    
    benchmark.run(show_plots=False, print_data=True)    