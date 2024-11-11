#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define M 256
#define K 512
#define N 256
#define BLOCK_SIZE 32

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

