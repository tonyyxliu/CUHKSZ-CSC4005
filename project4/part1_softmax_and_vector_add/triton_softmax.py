import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def native_softmax(x, mask=None, scale=1.0, dropout_p=0.0):

    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    x = x * scale
    if mask is not None:
        x = torch.where(mask, x, torch.tensor(-float('inf'), device=x.device))
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    if dropout_p > 0.0:
        ret = torch.nn.functional.dropout(ret, p=dropout_p, training=True)
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(output_ptr, 
                   input_ptr,
                   mask_ptr,
                   input_row_stride, 
                   output_row_stride, 
                   n_rows, 
                   n_cols,
                   scale,            
                   dropout_p,        
                   seed,               
                   HAS_DROPOUT: tl.constexpr,       
                   HAS_MASK: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # Your code here
    # TODO

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x, mask=None, scale=1.0, dropout_p=0.0):
    # Your code here
    # TODO
    

def torch_softmax(x, mask=None, scale=1.0, dropout_p=0.0):
    x = x * scale
    if mask is not None:
        x = torch.where(mask, x, torch.tensor(-float('inf'), device=x.device))
    y = torch.softmax(x, dim=-1)
    if dropout_p > 0.0:
        y = torch.nn.functional.dropout(y, p=dropout_p, training=True)
        
    return y

def unit_test(n_rows, n_cols):
    torch.manual_seed(0)
    x = torch.randn(n_rows, n_cols, device=DEVICE)
    mask = torch.rand_like(x) < 0.7
    mask[torch.arange(n_rows), torch.randint(0, n_cols, (n_rows,))] = True
    
    scale = 1.0 / (n_cols ** 0.5) # 典型的 scaling
    
    # Test 1: Correctness without Dropout
    y_triton = softmax(x, mask, scale=scale, dropout_p=0.0)
    y_torch = torch_softmax(x, mask, scale=scale, dropout_p=0.0)
    
    assert torch.allclose(y_triton, y_torch, atol=1e-5), "Mismatch with dropout=0"
    print(f"{n_rows} * {n_cols} [Scaling]: Correct!")

    # Test 2: Run with Dropout
    dropout_p = 0.5
    y_triton_drop = softmax(x, mask, scale=scale, dropout_p=dropout_p)
    
    zeros_fraction = (y_triton_drop == 0).float().mean().item()
    print(f"{n_rows} * {n_cols} [Dropout]: Ran successfully. Zeros fraction: {zeros_fraction:.2f}")
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[2 ** i for i in range(6, 14)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    mask = torch.rand_like(x) < 0.7
    mask[torch.arange(M), torch.randint(0, N, (M,))] = True
    
    scale = 0.5
    dropout_p = 0.1
    
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch_softmax(x, mask, scale, dropout_p))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x, mask, scale, dropout_p))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: native_softmax(x, mask, scale, dropout_p))
    
    return ms

if __name__ == '__main__':
    for i in range(8, 14):
        unit_test(2 ** i, 2 ** (i // 2))
    print("pass all correctness test")
    print("-----------------------------------------")    
    
    benchmark.run(show_plots=False, print_data=True)    