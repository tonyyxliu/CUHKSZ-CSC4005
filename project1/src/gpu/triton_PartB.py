import triton
import triton.language as tl
import triton.testing as tt
import numpy as np
import cv2
import torch


@triton.jit
def soft_blur_kernel(
    img_pad_ptr,  # *Pointer* to first input vector.
    pad_size,
    output_ptr,  # *Pointer* to output vector.
    sigma_blur,
    stride_h_pad,
    stride_w_pad,
    stride_h_out,
    stride_w_out,
    ACTIVATION: tl.constexpr,
):
    pid_h = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)

    offset = (
        (pid_h + pad_size) * stride_h_pad + (pid_w + pad_size) * stride_w_pad + pid_c
    )

    result = 0.0
    for sub_h in range(-pad_size, pad_size + 1):
        for sub_w in range(-pad_size, pad_size + 1):
            pixel_value = tl.load(
                img_pad_ptr + offset + sub_h * stride_h_pad + sub_w * stride_w_pad
            )
            result = result + pixel_value * sigma_blur

    output_offset = pid_h * stride_h_out + pid_w * stride_w_out + pid_c
    tl.store(output_ptr + output_offset, result)


def blur_filter(img_pad, k_size, activation=""):
    assert img_pad.is_contiguous(), "Matrix A must be contiguous"
    H, W, C = img_pad.shape
    sigma_blur = 1 / (k_size**2)  # 1/9 blur
    pad = (k_size - 1) // 2
    H_orig, W_orig = H - 2 * pad, W - 2 * pad  # ignore bounary pixels

    output = torch.empty(
        (H_orig, W_orig, C), device=img_pad.device, dtype=torch.float32
    )
    grid = lambda META: (
        triton.cdiv(H_orig, 1),
        triton.cdiv(W_orig, 1),
        triton.cdiv(C, 1),
    )

    elapsed_time = tt.do_bench(
        lambda: soft_blur_kernel[grid](
            img_pad,
            pad,
            output,
            sigma_blur,
            img_pad.stride(0),
            img_pad.stride(1),
            output.stride(0),
            output.stride(1),
            ACTIVATION=activation,
        ),
        warmup=25,  # time to warm up the kernel
        rep=100,
    )
    print(f"Execution Time: {elapsed_time:.2f} ms")
    return output


def main(input_image_path, output_image_path):
    # kernel size
    ksize = 3
    # read image
    print(f"Input file from: {input_image_path}")
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR).astype(np.float32)
    # add padding to the image
    pad = (ksize - 1) // 2

    pad_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    pad_img = torch.tensor(pad_img, device="cuda", dtype=torch.float32)

    # apply the filter
    output_triton = blur_filter(pad_img, ksize)
    output_img = output_triton.cpu().numpy()

    # save the output
    print(f"Output file to: {output_image_path}")
    # Transform pixel data from float32 to uint8 as image output
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_image_path, output_img)

    del output_triton
    del pad_img
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Invalid argument, should be: python3 script.py /path/to/input/jpeg /path/to/output/jpeg"
        )
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])
