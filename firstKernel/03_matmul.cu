#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define M 1024
#define K 2048
#define N 1024
#define BLOCK_SIZE 16
#define ITERATIONS 10

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

// initialize matrix
void init_matrix(float *matrix, int row, int col) {
    int size = row * col;
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// function to measure time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // allocate host memory
    h_a = (float*)malloc(size_A);
    h_b = (float*)malloc(size_B);
    h_c_cpu = (float*)malloc(size_C);
    h_c_gpu = (float*)malloc(size_C);

    // initialize matrix
    srand(time(NULL)); // seed random number generator
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);

    // allocate device memory
    cudaMalloc((void**)&d_a, size_A);
    cudaMalloc((void**)&d_b, size_B);
    cudaMalloc((void**)&d_c, size_C);

    // copy data from host to device
    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // warm up
    printf("Warming up...\n");
    for (int i = 0; i < ITERATIONS; i++) {
        matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
        matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, K, N);
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
        matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, K, N);
        cudaDeviceSynchronize();
        gpu_time += get_time() - start;
    }
    gpu_time /= ITERATIONS;
    printf("GPU time: %f ms\n", gpu_time * 1000);
    printf("Speedup: %f\n", cpu_time / gpu_time);

    // verify results
    printf("Verifying results...\n");
    int error_count = 0;
    cudaMemcpy(h_c_gpu, d_c, size_C, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_c_cpu[i] - h_c_gpu[i]);
        if (diff > 1e-3) {
            printf("Error at position %d: CPU=%f, GPU=%f\n", i, h_c_cpu[i], h_c_gpu[i]);
            printf("Difference: %f\n", diff);
            error_count++;
        }
    }
    if (error_count == 0) {
        printf("Results verified successfully!\n");
    } else {
        printf("Verification failed with %d errors.\n", error_count);
    }

    // free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// $ nvcc -o 03 03_matmul.cu
// $ ./03
// Warming up...
// Benchmarking CPU implementation...
// CPU time: 135.350180 ms
// Benchmarking GPU implementation...
// GPU time: 0.033693 ms
// Speedup: 4017.172162
// Verifying results...
// Results verified successfully!