#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000
#define BLOCK_SIZE 256

// CPU version addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

// intiailize vector with random values
void init_vector(float *vect, int n) {
    for (int i = 0; i < n; i++) {
        vect[i] = (float)rand() / RAND_MAX;
    }
}

// function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // allocate host memory
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c_cpu = (float *)malloc(size);
    h_c_gpu = (float *)malloc(size);

    // initilize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }
    // benchmark cpu implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // benchmark gpu implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_gpu<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // print results
    printf("CPU time: %f ms\n", cpu_avg_time * 1000);
    printf("GPU time: %f ms\n", gpu_avg_time * 1000);
    printf("Speedup: %f\n", cpu_avg_time / gpu_avg_time);

    // verify results
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            printf("Error at index %d: CPU=%f, GPU=%f\n", i, h_c_cpu[i], h_c_gpu[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Verification passed!\n");
    }

    // free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

// Output:
// Performing warm-up runs...
// Benchmarking CPU implementation...
// Benchmarking GPU implementation...
// CPU time: 62.652942 ms
// GPU time: 0.194492 ms
// Speedup: 322.136839
// Verification passed!