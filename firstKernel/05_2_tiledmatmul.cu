#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define TILE_SIZE_X 16
#define TILE_SIZE_Y 16

#define M 1024
#define K 2048
#define N 1024
#define ITERATIONS 5
#define WARMUP 2

// CPU matrix multiplication
void matmul_cpu(float *a, float *b, float *c, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = 0;
            for (int l = 0; l < k; l++) {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
};

// CUDA kernel for matrix multiplication
__global__ void matmul_gpu(float *a, float *b, float *c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
};

// CUDA kernel for tiled matrix multiplication
__global__ void tiled_matmul_kernel(float *a, float *b, float *c, int m, int k, int n) {

    // shared memory for a tile
    __shared__ float s_a[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ float s_b[TILE_SIZE_Y][TILE_SIZE_X];

    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // index of the first element in the tile
    int row = by * TILE_SIZE_Y + ty;
    int col = bx * TILE_SIZE_X + tx;

    float sum = 0.0f;

    for (int phase = 0; phase < (k + TILE_SIZE_X - 1) / TILE_SIZE_X; phase++) {
        // load elements from global memory to shared memory
        if (row < m && phase * TILE_SIZE_X + tx < k) {  
            s_a[ty][tx] = a[row * k + phase * TILE_SIZE_X + tx];
        }
        else {
            s_a[ty][tx] = 0;
        }

        if (col < n && phase * TILE_SIZE_Y + ty < k) {
            s_b[ty][tx] = b[(phase * TILE_SIZE_Y + ty) * n + col];
        }
        else {
            s_b[ty][tx] = 0;
        }
        __syncthreads();

        // compute the sum
        for (int i = 0; i < TILE_SIZE_X; i++) {
            sum += s_a[ty][i] * s_b[i][tx];
        }
        __syncthreads();
    }

    c[row * n + col] = sum;
}

// function to measure time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// initialize matrix
void init_matrix(float *matrix, int row, int col) {
    int size = row * col;
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

int main() {

    // Allocate memory on the host
    float *h_a = new float[M * K];
    float *h_b = new float[K * N];
    float *h_c_cpu = new float[M * N];
    float *h_c_gpu = new float[M * N];
    float *h_c_tiled = new float[M * N];

    // Initialize the matrices
    srand(42);
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);

    // Allocate memory on the device
    float *d_a, *d_b, *d_c_gpu, *d_c_tiled;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c_gpu, M * N * sizeof(float));
    cudaMalloc(&d_c_tiled, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 threadsPerBlock(TILE_SIZE_X, TILE_SIZE_Y);
    dim3 blocksPerGrid((N + TILE_SIZE_X - 1) / TILE_SIZE_X, (M + TILE_SIZE_Y - 1) / TILE_SIZE_Y);
    
    // warm up
    printf("Warming up...\n");
    for (int i = 0; i < WARMUP; i++) {
        matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
        matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c_gpu, M, K, N);
        cudaDeviceSynchronize();
        tiled_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c_tiled, M, K, N);
        cudaDeviceSynchronize();
    }

    // benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_time = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        double start = get_time();
        matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
        cpu_time += get_time() - start;
    }
    cpu_time /= ITERATIONS;
    printf("CPU time: %f ms\n", cpu_time * 1000);

    // benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_time = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        double start = get_time();
        matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c_gpu, M, K, N);
        cudaDeviceSynchronize();
        gpu_time += get_time() - start;
    }
    gpu_time /= ITERATIONS;
    printf("GPU time: %f ms\n", gpu_time * 1000);
    printf("Speedup GPU over CPU: %f\n", cpu_time / gpu_time);

    // verify results
    printf("Verifying results of GPU matmul...\n");
    int error_count = 0;
    cudaMemcpy(h_c_gpu, d_c_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_c_cpu[i] - h_c_gpu[i]); 
        if (diff > 1e-3) {
            printf("Error at position of GPU MatMul %d: CPU=%f, GPU=%f\n", i, h_c_cpu[i], h_c_gpu[i]);
            error_count++;
        }
    }
    if (error_count == 0) {
        printf("GPU Results verified successfully!\n");
    } else {
        printf("GPU Verification failed with %d errors.\n", error_count);
    }

    // benchmark tiled implementation
    printf("Benchmarking tiled implementation...\n");
    double tiled_time = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        double start = get_time();
        tiled_matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c_tiled, M, K, N);
        cudaDeviceSynchronize();
        tiled_time += get_time() - start;
    }
    tiled_time /= ITERATIONS;
    printf("Tiled time: %f ms\n", tiled_time * 1000);
    printf("Speedup Tiled over GPU: %f\n", gpu_time / tiled_time);

    // verify results
    printf("Verifying results of Tiled MatMul...\n");
    error_count = 0;
    cudaMemcpy(h_c_tiled, d_c_tiled, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_c_cpu[i] - h_c_tiled[i]);
        if (diff > 1e-3) {
            printf("Error at position of Tiled MatMul %d: CPU=%f, Tiled=%f\n", i, h_c_cpu[i], h_c_tiled[i]);
            error_count++;
        }
    }
    if (error_count == 0) {
        printf("Tiled Results verified successfully!\n");
    } else {
        printf("Tiled Verification failed with %d errors.\n", error_count);
    }

    // Free memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_gpu);
    cudaFree(d_c_tiled);

    // free memory on the host
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    delete[] h_c_tiled;

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    return 0;
}