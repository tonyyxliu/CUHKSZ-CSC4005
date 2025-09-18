import triton
import triton.language as tl
import triton.testing as tt
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# Triton kernel for RGB to Grayscale conversion
@triton.jit
def rgb_to_gray_kernel(
    buffer_ptr, gray_ptr, width, height, num_channels, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (width * height)

    # Calculate the index for each channel
    r = tl.load(buffer_ptr + idx * num_channels, mask=mask)
    g = tl.load(buffer_ptr + idx * num_channels + 1, mask=mask)
    b = tl.load(buffer_ptr + idx * num_channels + 2, mask=mask)

    # Convert to float32 for computation
    r = r.to(tl.float32)
    g = g.to(tl.float32)
    b = b.to(tl.float32)

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray_uint8 = gray.to(tl.uint8)  # Convert to uint8
    tl.store(gray_ptr + idx, gray_uint8, mask=mask)


def read_from_jpeg(filepath):
    image = Image.open(filepath)
    image = image.convert("RGB")
    image_array = torch.tensor(image.getdata(), dtype=torch.uint8).reshape(
        image.size[1], image.size[0], 3
    )
    return image_array, image.width, image.height, 3


def export_jpeg(image_tensor, filepath):
    image_array = image_tensor.byte().cpu()  # No need to permute for grayscale image
    image = Image.fromarray(image_array.numpy())
    image.save(filepath)


def main(input_filepath, output_filepath):
    # Read from input JPEG
    print(f"Input file from: {input_filepath}")
    input_image, width, height, num_channels = read_from_jpeg(input_filepath)

    # Allocate memory for grayscale image and buffer
    gray_image = torch.empty((height, width), dtype=torch.uint8)
    buffer = input_image.flatten()

    # Copy data to GPU
    buffer_ptr = buffer.cuda()
    gray_ptr = gray_image.flatten().cuda()

    grid = ((width * height + 512 - 1) // 512,)

    # Measure kernel execution time
    elapsed_time = tt.do_bench(
        lambda: rgb_to_gray_kernel[grid](
            buffer_ptr, gray_ptr, width, height, num_channels, BLOCK_SIZE=512
        ),
        warmup=25,  # time to warm up the kernel
        rep=100,
    )
    print(f"Execution Time: {elapsed_time:.2f} ms")

    # Copy result back to host
    gray_image = gray_ptr.cpu().reshape((height, width))

    # Write GrayImage to output JPEG
    print(f"Output file to: {output_filepath}")
    export_jpeg(gray_image, output_filepath)

    # Free GPU memory
    del buffer_ptr
    del gray_ptr
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Invalid argument, should be: python3 script.py /path/to/input/jpeg /path/to/output/jpeg"
        )
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])
