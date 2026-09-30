from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import time

# Load image
img = plt.imread("img.jpg")
h, w, _ = img.shape
print("image size:", w, "x", h)

# CPU grayscale
start_time = time.time()
gray_cpu = np.zeros((h, w, 3), dtype=np.uint8)
for y in range(h):
    for x in range(w):
        gray = (int(img[y, x, 0]) + int(img[y, x, 1]) + int(img[y, x, 2])) // 3
        gray_cpu[y, x, 0] = gray
        gray_cpu[y, x, 1] = gray
        gray_cpu[y, x, 2] = gray
cpu_time = time.time() - start_time
print("CPU time:", cpu_time, "s")
plt.imsave("gray_cpu.jpg", gray_cpu)

# GPU grayscale
@cuda.jit
def grayscale(src, dst):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if x < src.shape[1] and y < src.shape[0]:
        gray = (int(src[y, x, 0]) + int(src[y, x, 1]) + int(src[y, x, 2])) // 3
        dst[y, x, 0] = gray
        dst[y, x, 1] = gray
        dst[y, x, 2] = gray

def gray_gpu(blockSize):
    bx, by = blockSize
    gridSize = ((w + bx - 1) // bx, (h + by - 1) // by) 
    devSrc = cuda.to_device(img)
    devDst = cuda.device_array((h, w, 3), np.uint8)
    grayscale[gridSize, blockSize](devSrc, devDst)
    return devDst.copy_to_host()

# First run to warm up the GPU (JIT compilation)
gray_gpu((8, 8))

start = time.time()
result = gray_gpu((8, 8))
gpu_time = time.time() - start
print("GPU Time:", gpu_time, "s")
print("speedup:", round(cpu_time / gpu_time), "x")
plt.imsave("gray_gpu.jpg", result)

# Block size vs speedup (100 runs per block size with error bars)
blockSizes = [
    (8, 8),     # 64 threads
    (16, 8),    # 128 threads
    (16, 16),   # 256 threads
    (32, 8),    # 256 threads
    (16, 32),   # 512 threads
    (32, 16),   # 512 threads
    (32, 32)    # 1024 threads (max GPU)
]

moyennes = []
ecarts = []
labels = []

for bs in blockSizes:
    sp_list = []
    for _ in range(100):
        t0 = time.time()
        gray_gpu(bs)
        t_gpu = time.time() - t0
        sp_list.append(cpu_time / t_gpu)

    mean_sp = np.mean(sp_list)
    std_sp = np.std(sp_list)

    moyennes.append(mean_sp)
    ecarts.append(std_sp)

    label = f"{bs[0]}x{bs[1]}"
    labels.append(label)
    print(f"block size {label} : speedup = {mean_sp:.2f}x ± {std_sp:.2f}x")

plt.figure(figsize=(9, 5))
plt.errorbar(labels, moyennes, yerr=ecarts, marker="o", capsize=4, ecolor="red", color="blue")
plt.xlabel("2D Block Size (bx x by)")
plt.ylabel("Speedup (x)")
plt.title("2D Block Size vs Speedup (Mean ± Std Dev over 100 runs)")
plt.grid(True, alpha=0.3)
plt.savefig("blocksize_vs_speedup.png", dpi=150)
