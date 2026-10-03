from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import time

img = plt.imread("img.jpg")
h, w, _ = img.shape
n = h * w
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


@cuda.jit
def equalize(gray, out, lut):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < gray.shape[1] and y < gray.shape[0]:
        out[y, x] = lut[gray[y, x]]


def histo_gpu(devGray):
    local = cuda.to_device(np.zeros((h, 256), np.int32))
    histo = cuda.device_array(256, np.int32)
    local_histo[(h + 63) // 64, 64](devGray, local)
    sum_histo[1, 256](local, histo)
    return histo.copy_to_host()


def equalize_gpu(blockSize):
    bx, by = blockSize
    gridSize = ((w + bx - 1) // bx, (h + by - 1) // by)
    devSrc = cuda.to_device(img)
    devGray = cuda.device_array((h, w), np.uint8)
    grayscale[gridSize, blockSize](devSrc, devGray)

    histo = histo_gpu(devGray)
    p = histo / n
    c = np.cumsum(p)
    lut = np.round(c * 255).astype(np.uint8)

    devOut = cuda.device_array((h, w), np.uint8)
    equalize[gridSize, blockSize](devGray, devOut, cuda.to_device(lut))
    return devGray, devOut, histo


devGray, devOut, histo_before = equalize_gpu((16, 16))
histo_after = histo_gpu(devOut)
plt.imsave("gray.jpg", devGray.copy_to_host(), cmap="gray", vmin=0, vmax=255)
plt.imsave("equalized.jpg", devOut.copy_to_host(), cmap="gray", vmin=0, vmax=255)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(256), histo_before)
plt.title("before")
plt.subplot(1, 2, 2)
plt.bar(range(256), histo_after)
plt.title("after")
plt.savefig("histograms.png", dpi=150)

times = []
for k in range(20):
    start = time.time()
    equalize_gpu((16, 16))
    times.append(time.time() - start)
print(f"GPU time (gray + histogram + equalization): {np.mean(times) * 1000:.2f} ms")
