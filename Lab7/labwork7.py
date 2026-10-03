from numba import cuda
import numba
import numpy as np
import matplotlib.pyplot as plt
import time

img = plt.imread("img.jpg")
h, w, _ = img.shape
n = h * w
print("image size:", w, "x", h)

BLOCK = 256


@cuda.jit
def grayscale(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        dst[y, x] = (int(src[y, x, 0]) + int(src[y, x, 1]) + int(src[y, x, 2])) // 3


@cuda.jit
def reduce_naive(src_min, src_max, dst_min, dst_max):
    cache_min = cuda.shared.array(BLOCK, numba.float32)
    cache_max = cuda.shared.array(BLOCK, numba.float32)
    localtid = cuda.threadIdx.x
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if tid < src_min.shape[0]:
        cache_min[localtid] = src_min[tid]
        cache_max[localtid] = src_max[tid]
    else:
        cache_min[localtid] = 255
        cache_max[localtid] = 0
    cuda.syncthreads()

    s = 1
    while s < cuda.blockDim.x:
        if localtid % (s * 2) == 0:
            cache_min[localtid] = min(cache_min[localtid], cache_min[localtid + s])
            cache_max[localtid] = max(cache_max[localtid], cache_max[localtid + s])
        cuda.syncthreads()
        s = s * 2

    if localtid == 0:
        dst_min[cuda.blockIdx.x] = cache_min[0]
        dst_max[cuda.blockIdx.x] = cache_max[0]


@cuda.jit
def reduce_opt(src_min, src_max, dst_min, dst_max):
    cache_min = cuda.shared.array(BLOCK, numba.float32)
    cache_max = cuda.shared.array(BLOCK, numba.float32)
    localtid = cuda.threadIdx.x
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x * 2
    size = src_min.shape[0]

    mn = 255.0
    mx = 0.0
    if tid < size:
        mn = src_min[tid]
        mx = src_max[tid]
    if tid + cuda.blockDim.x < size:
        mn = min(mn, src_min[tid + cuda.blockDim.x])
        mx = max(mx, src_max[tid + cuda.blockDim.x])
    cache_min[localtid] = mn
    cache_max[localtid] = mx
    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    while s > 0:
        if localtid < s:
            cache_min[localtid] = min(cache_min[localtid], cache_min[localtid + s])
            cache_max[localtid] = max(cache_max[localtid], cache_max[localtid + s])
        cuda.syncthreads()
        s = s // 2

    if localtid == 0:
        dst_min[cuda.blockIdx.x] = cache_min[0]
        dst_max[cuda.blockIdx.x] = cache_max[0]


@cuda.jit
def stretch(src, dst, mn, mx):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        dst[y, x] = (src[y, x] - mn) / (mx - mn) * 255


def find_min_max(kernel, devGray, per_block):
    blocks = (n + per_block - 1) // per_block
    out_min = cuda.device_array(blocks, np.float32)
    out_max = cuda.device_array(blocks, np.float32)
    data = devGray.reshape(n)
    kernel[blocks, BLOCK](data, data, out_min, out_max)
    return out_min.copy_to_host().min(), out_max.copy_to_host().max()


def gray_stretch(blockSize):
    bx, by = blockSize
    gridSize = ((w + bx - 1) // bx, (h + by - 1) // by)
    devSrc = cuda.to_device(img)
    devGray = cuda.device_array((h, w), np.uint8)
    devOut = cuda.device_array((h, w), np.uint8)
    grayscale[gridSize, blockSize](devSrc, devGray)
    mn, mx = find_min_max(reduce_opt, devGray, BLOCK * 2)
    stretch[gridSize, blockSize](devGray, devOut, mn, mx)
    return devGray, devOut.copy_to_host(), mn, mx





# Results 

devGray, result, mn, mx = gray_stretch((16, 16))
print("min:", mn, "max:", mx)
plt.imsave("gray.jpg", devGray.copy_to_host(), cmap="gray", vmin=0, vmax=255)
plt.imsave("stretched.jpg", result, cmap="gray", vmin=0, vmax=255)


find_min_max(reduce_naive, devGray, BLOCK)
times_naive = []
times_opt = []


for k in range(20):
    start = time.time()
    find_min_max(reduce_naive, devGray, BLOCK)
    times_naive.append(time.time() - start)

    start = time.time()
    find_min_max(reduce_opt, devGray, BLOCK * 2)
    times_opt.append(time.time() - start)



print(f"reduction naive: {np.mean(times_naive)*1000:.3f} ms, optimized: {np.mean(times_opt)*1000:.3f} ms")
print(f"speedup of the optimization: {np.mean(times_naive) / np.mean(times_opt):.2f}x")




blockSizes = [(8, 8), (16, 8), (16, 16), (32, 8), (16, 32), (32, 16), (32, 32)]
for bs in blockSizes:
    times = []
    for k in range(20):
        start = time.time()
        gray_stretch(bs)
        times.append((time.time() - start) * 1000)
    print(f"block size {bs[0]}x{bs[1]}: {np.mean(times):.2f} +/- {np.std(times):.2f} ms")
