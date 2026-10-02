from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Load image
img = plt.imread("img.jpg")
h, w, _ = img.shape
print("image size:", w, "x", h)

value = int(sys.argv[1]) if len(sys.argv) > 1 else 50

blockSizes = [(8, 8), (16, 8), (16, 16), (32, 8), (16, 32), (32, 16), (32, 32)]


@cuda.jit
def brightness(src, dst, value):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        for c in range(3):
            new = int(src[y, x, c]) + value
            if new > 255: new = 255
            if new < 0: new = 0
            dst[y, x, c] = new



def brightness_gpu(blockSize, value):
    bx, by = blockSize
    gridSize = ((w + bx - 1) // bx, (h + by - 1) // by)
    devSrc = cuda.to_device(img)
    devDst = cuda.device_array((h, w, 3), np.uint8)
    brightness[gridSize, blockSize](devSrc, devDst, value)
    return devDst.copy_to_host()

plt.imsave("brighter.jpg", brightness_gpu((16, 16), value))
plt.imsave("darker.jpg", brightness_gpu((16, 16), -value))


for bs in blockSizes:
    times = []
    for _ in range(20):
        start = time.time()
        brightness_gpu(bs, value)
        end = time.time()
        times.append((end - start) * 1000)  # convert to ms
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Block size: {bs}, Mean time: {mean_time:.2f} ms, Std: {std_time:.2f} ms")

