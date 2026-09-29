from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import time

# Load image
img = plt.imread("img.jpg")
h, w, _ = img.shape
print("image size:", w, "x", h)

# Flatten
pixelCount = h * w
flat = img.reshape((pixelCount, 3))


# CPU grayscale
start = time.time()
gray_cpu = np.zeros((pixelCount, 3), np.uint8)
for i in range(pixelCount):
    g = (int(flat[i, 0]) + int(flat[i, 1]) + int(flat[i, 2])) // 3
    gray_cpu[i, 0] = g
    gray_cpu[i, 1] = g
    gray_cpu[i, 2] = g
cpu_time = time.time() - start
print("CPU time:", cpu_time, "s")
plt.imsave("gray_cpu.jpg", gray_cpu.reshape((h, w, 3)))


# GPU grayscale
@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx < src.shape[0]:
        g = (int(src[tidx, 0]) + int(src[tidx, 1]) + int(src[tidx, 2])) // 3
        dst[tidx, 0] = g
        dst[tidx, 1] = g
        dst[tidx, 2] = g

def gray_gpu(blockSize):
    gridSize = (pixelCount + blockSize - 1) // blockSize
    devSrc = cuda.to_device(flat)
    devDst = cuda.device_array((pixelCount, 3), np.uint8)
    grayscale[gridSize, blockSize](devSrc, devDst)
    return devDst.copy_to_host()

gray_gpu(64)  # first run is just to compile

start = time.time()
result = gray_gpu(64)
gpu_time = time.time() - start
print("GPU time:", gpu_time, "s")
print("speedup:", round(cpu_time / gpu_time), "x")
plt.imsave("gray_gpu.jpg", result.reshape((h, w, 3)))


# Block size vs time
blockSizes = [32, 64, 128, 256, 512, 1024]
times = []
for bs in blockSizes:
    start = time.time()
    for k in range(20):
        gray_gpu(bs)
    t = (time.time() - start) / 20 * 1000
    times.append(t)
    print("block size", bs, ":", round(t, 2), "ms")

plt.figure(figsize=(8, 5))
plt.plot(blockSizes, times, marker="o")
plt.xscale("log", base=2)
plt.xticks(blockSizes, blockSizes)
plt.ylim(0, max(times) * 1.2)
plt.xlabel("block size")
plt.ylabel("time (ms)")
plt.title("block size vs time")
plt.grid(True, alpha=0.3)
plt.savefig("blocksize_vs_time.png", dpi=150)
