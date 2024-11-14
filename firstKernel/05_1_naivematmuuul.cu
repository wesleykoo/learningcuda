#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul_kernel(float *a, float *b, float *c, int m, int k, int n) {
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

int main() {
    // define matrix sizes
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    // calculate matrix sizes in bytes
    const size_t a_size = M * K * sizeof(float);
    const size_t b_size = K * N * sizeof(float);
    const size_t c_size = M * N * sizeof(float);

    // declare device pointers
    float *d_a, *d_b, *d_c;

    // declare host pointers
    float *h_a = new float[M * K];
    float *h_b = new float[K * N];
    float *h_c = new float[M * N];

    // allocate memory on the device
    cudaMalloc(&d_a, a_size);
    cudaMalloc(&d_b, b_size);
    cudaMalloc(&d_c, c_size);

    // launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();

    // copy result back to host
    cudaMemcpy(h_c, d_c, c_size, cudaMemcpyDeviceToHost);

    // free memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // print result
    std::cout << "Result: " << h_c[0] << std::endl;

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    return 0;
}