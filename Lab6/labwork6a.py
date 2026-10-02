from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Load image
img = plt.imread("img.jpg")
h, w, _ = img.shape
print("image size:", w, "x", h)

threshold = int(sys.argv[1]) if len(sys.argv) > 1 else 128


@cuda.jit
def grayscale(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        gray = (int(src[y, x, 0]) + int(src[y, x, 1]) + int(src[y, x, 2])) // 3
        dst[y, x, 0] = gray
        dst[y, x, 1] = gray
        dst[y, x, 2] = gray


blockSizes = [(8, 8), (16, 8), (16, 16), (32, 8), (16, 32), (32, 16), (32, 32)]

@cuda.jit
def binarize(src, dst, threshold):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        value = 0 if src[y, x, 0] < threshold else 255
        dst[y, x, 0] = value
        dst[y, x, 1] = value
        dst[y, x, 2] = value


def binarize_gpu(blockSize):
    bx, by = blockSize
    gridSize = ((w + bx - 1) // bx, (h + by - 1) // by)
    devSrc = cuda.to_device(img)
    devGray = cuda.device_array((h, w, 3), np.uint8)
    devBin  = cuda.device_array((h, w, 3), np.uint8)
    grayscale[gridSize, blockSize](devSrc, devGray)
    binarize[gridSize, blockSize](devGray, devBin, threshold)
    return devBin.copy_to_host()

result = binarize_gpu((16, 16))
plt.imsave("binarized.jpg", result)


for bs in blockSizes:
    times = []
    for _ in range(20):
        start = time.time()
        binarize_gpu(bs)
        end = time.time()
        times.append((end - start) * 1000)  # convert to ms
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Block size {bs}: mean = {mean_time:.2f} ms, std = {std_time:.2f} ms")
