import numpy as np
import time
from numba import cuda
import matplotlib.pyplot as plt

# Функция для перемножения матриц на CPU
def matrix_multiply_cpu(A, B):
    return np.dot(A, B)

# Функция для перемножения матриц на GPU
@cuda.jit
def matrix_multiply_gpu(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def main():
    sizes = [100, 500, 1000, 1500, 2000]
    cpu_times = []
    gpu_times = []
    speedups = []

    for size in sizes:
        A = np.random.randint(0, 10, size=(size, size)).astype(np.int32)
        B = np.random.randint(0, 10, size=(size, size)).astype(np.int32)

        # CPU
        start_time = time.time()
        C_cpu = matrix_multiply_cpu(A, B)
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)

        # GPU
        A_gpu = cuda.to_device(A)
        B_gpu = cuda.to_device(B)
        C_gpu = cuda.device_array((size, size), dtype=np.int32)

        threads_per_block = (16, 16)
        blocks_per_grid_x = int(np.ceil(size / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(size / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        start_time = time.time()
        matrix_multiply_gpu[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)
        C_gpu.copy_to_host()
        gpu_time = time.time() - start_time
        gpu_times.append(gpu_time)

        # Проверка корректности
        assert np.allclose(C_cpu, C_gpu), "Results do not match!"

        # Ускорение
        speedup = cpu_time / gpu_time
        speedups.append(speedup)

        print(f"Matrix size = {size}x{size}\t CPU time: {cpu_time:.4f}\t GPU time: {gpu_time:.4f}\tSpeedup: {speedup:.2f}")


    # Построение графиков
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, cpu_times, label='CPU Time', marker='o')
    plt.plot(sizes, gpu_times, label='GPU Time', marker='o')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (s)')
    plt.title('Time vs Matrix Size')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(sizes, speedups, label='Speedup (CPU/GPU)', marker='o')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Matrix Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()