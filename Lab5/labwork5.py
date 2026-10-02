from numba import cuda
import numba
import numpy as np
import matplotlib.pyplot as plt
import time

# Load image
img = plt.imread("img.jpg")
h, w, _ = img.shape
print("image size:", w, "x", h)


# Gaussian filter
sigma = 1.0
gaussian = np.zeros((7, 7), dtype=np.float32)
for y in range(7):
    for x in range(7):
        gaussian[y, x] = 1 / (2 * np.pi * sigma**2) * np.exp(-((x - 3)**2 + (y - 3)**2) / (2 * sigma**2))
gaussian = gaussian / gaussian.sum()
print(np.round(gaussian * 1000).astype(int))



# CPU (on a crop)
def cpu_blur(img):
    h, w, _ = img.shape
    out = img.copy()
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            for c in range(3):
                s = 0.0
                for j in range(7):
                    for i in range(7):
                        s += img[y + j - 3, x + i - 3, c] * gaussian[j, i]
                out[y, x, c] = s
    return out

crop = img[1300:1556, 1800:2056]
start = time.time()
cpu_result = cpu_blur(crop)
crop_time = time.time() - start
cpu_time = crop_time * (h * w) / (256 * 256)
print("CPU time crop:", crop_time, "s")
print("CPU time full image (estimated):", cpu_time, "s")
plt.imsave("crop_original.jpg", crop)
plt.imsave("crop_blur_cpu.jpg", cpu_result)


# GPU without shared memory
@cuda.jit
def blur(src, dst, filter):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < 3 or y < 3 or x >= src.shape[1] - 3 or y >= src.shape[0] - 3:
        return
    for c in range(3):
        s = 0.0
        for j in range(7):
            for i in range(7):
                s += src[y + j - 3, x + i - 3, c] * filter[j, i]
        dst[y, x, c] = s


# GPU with shared memory
@cuda.jit
def blur_shared(src, dst, filter):
    tile = cuda.shared.array((7, 7), numba.float32)
    if cuda.threadIdx.x < 7 and cuda.threadIdx.y < 7:
        tile[cuda.threadIdx.y, cuda.threadIdx.x] = filter[cuda.threadIdx.y, cuda.threadIdx.x]
    cuda.syncthreads()

    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < 3 or y < 3 or x >= src.shape[1] - 3 or y >= src.shape[0] - 3:
        return
    for c in range(3):
        s = 0.0
        for j in range(7):
            for i in range(7):
                s += src[y + j - 3, x + i - 3, c] * tile[j, i]
        dst[y, x, c] = s


def blur_gpu(kernel, blockSize):
    gridSize = ((w + blockSize[0] - 1) // blockSize[0], (h + blockSize[1] - 1) // blockSize[1])
    devSrc = cuda.to_device(img)
    devDst = cuda.to_device(img)
    devFilter = cuda.to_device(gaussian)
    kernel[gridSize, blockSize](devSrc, devDst, devFilter)
    return devDst.copy_to_host()

# first run (compilation)
plt.imsave("blur_gpu.jpg", blur_gpu(blur, (16, 16)))
plt.imsave("blur_gpu_shared.jpg", blur_gpu(blur_shared, (16, 16)))


# Block size vs speedup
blockSizes = [(8, 8), (16, 8), (16, 16), (32, 8), (16, 32), (32, 16), (32, 32)]
labels = []
means = []
stds = []
means_shared = []
stds_shared = []

for bs in blockSizes:
    sp = []
    sp_shared = []
    for k in range(20):
        start = time.time()
        blur_gpu(blur, bs)
        sp.append(cpu_time / (time.time() - start))

        start = time.time()
        blur_gpu(blur_shared, bs)
        sp_shared.append(cpu_time / (time.time() - start))

    labels.append(f"{bs[0]}x{bs[1]}")
    means.append(np.mean(sp))
    stds.append(np.std(sp))
    means_shared.append(np.mean(sp_shared))
    stds_shared.append(np.std(sp_shared))
    print(f"{bs[0]}x{bs[1]}: without shared {np.mean(sp):.0f}x ± {np.std(sp):.0f}, with shared {np.mean(sp_shared):.0f}x ± {np.std(sp_shared):.0f}")

plt.figure(figsize=(9, 5))
plt.errorbar(labels, means, yerr=stds, marker="o", capsize=4, label="without shared memory")
plt.errorbar(labels, means_shared, yerr=stds_shared, marker="o", capsize=4, label="with shared memory")
plt.xlabel("block size")
plt.ylabel("speedup")
plt.title("block size vs speedup")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("blocksize_vs_speedup.png", dpi=150)
