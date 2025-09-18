import torch
import torch.nn.functional as F
import triton.testing as tt
import triton
import triton.language as tl
import numpy as np
import cv2


@triton.jit
def bilateral_filter_kernel(
    img_pad_ptr,  # *Pointer* to first input vector.
    height,
    width,
    channel,
    pad_size,
    output_ptr,
    stride_h_pad,
    stride_w_pad,
    stride_h_out,
    stride_w_out,
    sigma_density,
    sigma_space,
    ACTIVATION: tl.constexpr,
):

    pid_h = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)

    # TODO: Triton PartC bilateral filter kernel.
    # You can write this part like the one in PartB


def main(input_image_path, output_image_path):
    print(f"Input file to: {input_image_path}")

    sigma_space = 1.7
    sigma_density = 50.0
    ksize = 3  # ksize=7

    # read image
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR).astype(np.float32)

    height, width, channel = img.shape
    # add padding to the image
    pad = (ksize - 1) // 2

    # TODO:
    # Triton PartC main body.
    # You can write this part like the one in PartB
    # Note: the time measurement should use tt.do_bench(kernel), just like the one in PartA & B


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Invalid argument, should be: python3 script.py /path/to/input/jpeg /path/to/output/jpeg"
        )
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])
