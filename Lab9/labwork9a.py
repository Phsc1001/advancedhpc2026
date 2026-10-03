from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import time

img = plt.imread("img.jpg")
h, w, _ = img.shape
print("image size:", w, "x", h)


@cuda.jit
def grayscale(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        dst[y, x] = (int(src[y, x, 0]) + int(src[y, x, 1]) + int(src[y, x, 2])) // 3


@cuda.jit
def local_histo(gray, local):
    y = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if y < gray.shape[0]:
        for x in range(gray.shape[1]):
            local[y, gray[y, x]] += 1


@cuda.jit
def sum_histo(local, histo):
    j = cuda.threadIdx.x
    total = 0
    for y in range(local.shape[0]):
        total += local[y, j]
    histo[j] = total


def gray_gpu(blockSize):
    bx, by = blockSize
    gridSize = ((w + bx - 1) // bx, (h + by - 1) // by)
    devSrc = cuda.to_device(img)
    devGray = cuda.device_array((h, w), np.uint8)
    grayscale[gridSize, blockSize](devSrc, devGray)
    return devGray


def histo_gpu(devGray):
    local = cuda.to_device(np.zeros((h, 256), np.int32))
    histo = cuda.device_array(256, np.int32)
    local_histo[(h + 63) // 64, 64](devGray, local)
    sum_histo[1, 256](local, histo)
    return histo.copy_to_host()


devGray = gray_gpu((16, 16))
histo = histo_gpu(devGray)
gray = devGray.copy_to_host()
print("correct:", np.array_equal(histo, np.bincount(gray.ravel(), minlength=256)), "| total:", histo.sum(), "=", h * w)

plt.bar(range(256), histo)
plt.xlabel("gray level")
plt.ylabel("number of pixels")
plt.title("histogram")
plt.savefig("histogram.png", dpi=150)


start = time.time()
cpu_histo = np.zeros(256, np.int64)
for y in range(200):
    for x in range(w):
        cpu_histo[gray[y, x]] += 1
cpu_time = (time.time() - start) * h / 200

times = []
for k in range(20):
    start = time.time()
    histo_gpu(devGray)
    times.append(time.time() - start)
gpu_time = np.mean(times)

print(f"CPU time (estimated): {cpu_time:.2f} s")
print(f"GPU time: {gpu_time * 1000:.2f} ms")
print(f"speedup: {cpu_time / gpu_time:.0f}x")
