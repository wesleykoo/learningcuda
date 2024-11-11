#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 10000000 // number of elements in the vector
#define BLOCK_SIZE_1D 1024 // number of threads per block
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8
#define NUM_ITERATIONS 10

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for 1D vector addition
__global__ void vector_add_gpu_1D(float *a, float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

// CUDA kernel for 3D vector addition
__global__ void vector_add_gpu_3D(float *a, float *b, float *c, int nx, int ny, int nz) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < nx && y < ny && z < nz) {
        int index = x + y * nx + z * nx * ny;
        if (index < nx * ny * nz) {
            c[index] = a[index] + b[index];
        }
    }
}

// initialize vector with randome values
void init_vector(float *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = (float)rand() / RAND_MAX;
    }
}

// function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1D, *h_c_gpu_3D;
    float *d_a, *d_b, *d_c_1D, *d_c_3D;
    size_t size = N * sizeof(float);

    // allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu_1D = (float*)malloc(size);
    h_c_gpu_3D = (float*)malloc(size);

    // initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1D, size);
    cudaMalloc(&d_c_3D, size);

    // copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // kernel launch parameters for 1D vector addition
    dim3 block_size_1D(BLOCK_SIZE_1D);
    dim3 grid_size_1D((N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);

    // kernel launch parameters for 3D vector addition
    int nx = 100, ny = 100, nz = 1000;
    dim3 block_size_3D(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 grid_size_3D(
        (nx + block_size_3D.x - 1) / block_size_3D.x,
        (ny + block_size_3D.y - 1) / block_size_3D.y,
        (nz + block_size_3D.z - 1) / block_size_3D.z
    );

    // warm-up runs
    printf("Warm-up runs...\n");
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu_1D<<<grid_size_1D, block_size_1D>>>(d_a, d_b, d_c_1D, N);
        vector_add_gpu_3D<<<grid_size_3D, block_size_3D>>>(d_a, d_b, d_c_3D, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    // benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_time = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        cpu_time += get_time() - start_time;
    }
    cpu_time /= NUM_ITERATIONS;
    printf("CPU time: %f ms\n", cpu_time * 1000);

    // benchmark GPU 1D implementation
    printf("Benchmarking GPU 1D implementation...\n");
    double gpu_1D_time = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        double start_time = get_time();
        vector_add_gpu_1D<<<grid_size_1D, block_size_1D>>>(d_a, d_b, d_c_1D, N);
        gpu_1D_time += get_time() - start_time;
    }
    gpu_1D_time /= NUM_ITERATIONS;
    printf("GPU 1D time: %f ms\n", gpu_1D_time * 1000);

    // verify results from GPU 1D implementation
    cudaMemcpy(h_c_gpu_1D, d_c_1D, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_gpu_1D[i] - h_c_cpu[i]) > 1e-5) {
            printf("Error: GPU 1D implementation is incorrect at index %d\n", i);
        }
    }

    // benchmark GPU 3D implementation
    printf("Benchmarking GPU 3D implementation...\n");
    double gpu_3D_time = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        double start_time = get_time();
        vector_add_gpu_3D<<<grid_size_3D, block_size_3D>>>(d_a, d_b, d_c_3D, nx, ny, nz);
        gpu_3D_time += get_time() - start_time;
    }
    gpu_3D_time /= NUM_ITERATIONS;
    printf("GPU 3D time: %f ms\n", gpu_3D_time * 1000); 

    // verify results from GPU 3D implementation
    cudaMemcpy(h_c_gpu_3D, d_c_3D, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_gpu_3D[i] - h_c_cpu[i]) > 1e-5) {
            printf("Error: GPU 3D implementation is incorrect at index %d\n", i);
        }
    }

    // print results
    printf("CPU average time: %f ms\n", cpu_time * 1000);
    printf("GPU 1D average time: %f ms\n", gpu_1D_time * 1000);
    printf("GPU 3D average time: %f ms\n", gpu_3D_time * 1000);
    printf("Speedup CPU vs GPU 1D: %f\n", cpu_time / gpu_1D_time);
    printf("Speedup CPU vs GPU 3D: %f\n", cpu_time / gpu_3D_time);
    printf("Speedup GPU 1D vs GPU 3D: %f\n", gpu_1D_time / gpu_3D_time);

    // free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1D);
    free(h_c_gpu_3D);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1D);
    cudaFree(d_c_3D);

    return 0;
}

// Output:
// Warm-up runs...
// Benchmarking CPU implementation...
// CPU time: 63.581947 ms
// Benchmarking GPU 1D implementation...
// GPU 1D time: 0.013040 ms
// Benchmarking GPU 3D implementation...
// GPU 3D time: 0.011479 ms
// CPU average time: 63.581947 ms
// GPU 1D average time: 0.013040 ms
// GPU 3D average time: 0.011479 ms
// Speedup CPU vs GPU 1D: 4875.916182
// Speedup CPU vs GPU 3D: 5539.220628
// Speedup GPU 1D vs GPU 3D: 1.136037