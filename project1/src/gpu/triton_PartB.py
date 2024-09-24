import triton
import triton.language as tl
import numpy as np
import cv2
import torch
import time

@triton.jit
def soft_blur_kernel(img_pad_ptr,  # *Pointer* to first input vector.
                    pad_size,
                    output_ptr,  # *Pointer* to output vector.
                    sigma_blur,
                    stride_h_pad, stride_w_pad,
                    stride_h_out, stride_w_out,
                    ACTIVATION: tl.constexpr
                   ):
    pid_h = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)
    offset = (pid_h + pad_size) * stride_h_pad + (pid_w + pad_size) * stride_w_pad
    result = 0.
    for sub_h in range(-pad_size, pad_size+1):
        for sub_w in range(-pad_size, pad_size+1):
            pixel_value = tl.load(img_pad_ptr + offset + sub_h * stride_h_pad + sub_w * stride_w_pad)
            result = result + pixel_value * sigma_blur
    output_offset = pid_h * stride_h_out + pid_w * stride_w_out
    tl.store(output_ptr + output_offset, result)

def blur_filter(img_pad, k_size, activation=""):
    assert img_pad.is_contiguous(), "Matrix A must be contiguous"
    H, W = img_pad.shape
    sigma_blur = 1 / (k_size ** 2)
    pad = (k_size-1) // 2
    H_orig, W_orig = H - 2*pad, W - 2*pad
    output = torch.empty((H_orig, W_orig), device=img_pad.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(H_orig, 1), triton.cdiv(W_orig, 1))
    soft_blur_kernel[grid](
        img_pad, pad, output, sigma_blur, img_pad.stride(0), img_pad.stride(1), output.stride(0), output.stride(1),ACTIVATION=activation
    )
    return output

def main(input_image_path, output_image_path, output_time_image_path):
    '''
    Code for image processing
    '''
    # kernel size
    ksize = 3
    # read image
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # add padding to the image
    pad = (ksize-1) // 2
    pad_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    pad_img = torch.tensor(pad_img, device="cuda", dtype=torch.float32)
    # apply the filter
    output_triton = blur_filter(pad_img, ksize)
    # save the output
    cv2.imwrite(output_image_path, output_triton.cpu().numpy())

    '''
    Code for time comparison
    '''
    time_list = []
    for i in range(50):
        # random image tensor
        # You can change the size of the image tensor to see the difference in time
        random_img = torch.rand(20000, 20000, dtype=torch.float32, device="cuda") * 255
        start = time.time()
        output_triton = blur_filter(random_img, ksize)
        end = time.time()
        time_list.append(end - start)
    # print the time list for result
    # print(time_list)
    import matplotlib.pyplot as plt
    # draw the time list in a graph, you can uncomment the line below to see the graph from 1-100 iterations
    plt.plot(time_list, label='triton')
    # draw the time list in a graph, you can uncomment the line below to see the graph from 10-100 iterations
    # plt.plot(time_list[10:], label='triton')
    plt.legend()
    plt.savefig(output_time_image_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Invalid argument, should be: python3 script.py /path/to/input/jpeg /path/to/output/jpeg_1 /path/to/output/jpeg_2")
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
