from numba import cuda

#Step 1 - List the GPUs
cuda.detect()  


#Step 2 - Select a GPUa
cuda.select_device(0)


#Step 3 - Device name and id
device = cuda.get_current_device() 
print(f"Device name: {device.name}")
device_id = device.id
print(f"Device id: {device_id}")


#Step 4 - Multiprocessor count (SM)
MPC = device.MULTIPROCESSOR_COUNT
print(f"Multiprocessor count: {MPC}")


#Step 5 - Core count
cores_per_sm = 128
total_cores = MPC * cores_per_sm
print(f"Total cores: {total_cores}")


#Step 6 - Memory size
free, total = cuda.current_context().get_memory_info()
print(f"Memory size: {total / 1024**3:.2f} GB")


#Step 7 - Extra info for the report
clock_rate = device.CLOCK_RATE
print(f"Clock rate: {clock_rate / 1000:.2f} MHz")
max_threads_per_block = device.MAX_THREADS_PER_BLOCK
print(f"Max threads per block: {max_threads_per_block}")
warp_size = device.WARP_SIZE
print(f"Warp size: {warp_size}")
memory_clock = device.MEMORY_CLOCK_RATE
print(f"Memory clock: {memory_clock / 1000:.2f} MHz")

