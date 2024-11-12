#include <nvtx3/nvtx3.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define ITERATIONS 1
#define BLOCK_SIZE 16

__global__ void matmul_kernel(float *a, float *b, float *c, int m, int k, int n) {
    
    // calculate row and column of the element to compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (row < m && col < n) {
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

void matmul_gpu(float *a, float *b, float *c, int m, int k, int n) {
    float *d_a, *d_b, *d_c;
    
    // allocate memory on the device
    nvtxRangePush("Allocate memory on the device");
    cudaMalloc(&d_a, m * k * sizeof(float));
    cudaMalloc(&d_b, k * n * sizeof(float));
    cudaMalloc(&d_c, m * n * sizeof(float));
    nvtxRangePop();

    
}