#include <nvtx3/nvToolsExt.h>
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
    nvtxRangePush("matmul_gpu");
    
    float *d_a, *d_b, *d_c;
    
    // allocate memory on the device
    nvtxRangePush("Allocate memory on the device");
    cudaMalloc(&d_a, m * k * sizeof(float));
    cudaMalloc(&d_b, k * n * sizeof(float));
    cudaMalloc(&d_c, m * n * sizeof(float));
    nvtxRangePop();

    // copy data from host to device
    nvtxRangePush("Copy data from host to device");
    cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);
    nvtxRangePop();

    // launch kernel
    nvtxRangePush("Launch kernel");
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, m, k, n);
    cudaDeviceSynchronize();
    nvtxRangePop();

    // copy data from device to host
    nvtxRangePush("Copy data from device to host");
    cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    nvtxRangePop();

    // free memory on the device
    nvtxRangePush("Free memory on the device");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    nvtxRangePop();

    nvtxRangePop();
}

int main() {
    const int N = 1024;
    float *h_a = new float[N * N];
    float *h_b = new float[N * N];
    float *h_c = new float[N * N];

    matmul_gpu(h_a, h_b, h_c, N, N, N);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}