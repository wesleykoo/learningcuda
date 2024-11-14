#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(float *a, float *b, float *c, int m, int k, int n) {

    // shared memory for a tile
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];

    // block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // index of the first element in the tile
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0;

    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // load elements from global memory to shared memory
        s_a[ty][tx] = a[row][tile * TILE_SIZE + tx];
        s_b[ty][tx] = b[(tile * TILE_SIZE + ty)][col];
    }



}



int main() {
    return 0;
}