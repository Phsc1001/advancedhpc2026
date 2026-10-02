from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Load images (same size)
img = plt.imread("img.jpg")
img2 = plt.imread("img2.jpg")
h, w, _ = img.shape
print("image size:", w, "x", h)

c = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

blockSizes = [(8, 8), (16, 8), (16, 16), (32, 8), (16, 32), (32, 16), (32, 32)]



@cuda.jit
def blend(src1, src2, dst, c):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src1.shape[1] and y < src1.shape[0]:
        for ch in range(3):
            dst[y, x, ch] = c * src1[y, x, ch] + (1 - c) * src2[y, x, ch]


def blend_gpu(blockSize, c):
    bx, by = blockSize
    gridSize = ((w + bx - 1) // bx, (h + by - 1) // by)
    devSrc1 = cuda.to_device(img)
    devSrc2 = cuda.to_device(img2)
    devDst = cuda.device_array((h, w, 3), np.uint8)
    blend[gridSize, blockSize](devSrc1, devSrc2, devDst, c)
    return devDst.copy_to_host()

plt.imsave("blend.jpg", blend_gpu((16, 16), c))
yanis = 9



for bs in blockSizes:
    times = []
    for _ in range(20):
        start = time.time()
        blend_gpu(bs, c)
        end = time.time()
        times.append((end - start) * 1000)  # convert to ms
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Block size: {bs}, Mean time: {mean_time:.2f} ms, Std: {std_time:.2f} ms")
