#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


__global__ void add_kernel(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
    //printf("%d %f %f %f\n", idx, a[idx], b[idx], c[idx]);
}

__global__ void single_kernel(float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    b[idx] = a[idx];
}

int main() {
    int size = 1024 * 1024;

    float* ha = (float*)malloc(size * sizeof(float));
    float* hb = (float*)malloc(size * sizeof(float));
    float* hc = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        ha[i] = i;
        hb[i] = i;
    }

    float* da;
    float* db;
    float* dc;
    cudaMalloc(&da, size * sizeof(float));
    cudaMalloc(&db, size * sizeof(float));
    cudaMalloc(&dc, size * sizeof(float));
    cudaMemcpy(da, ha, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size * sizeof(float), cudaMemcpyHostToDevice);

    int grid = 1;
    int block = 512;
    for (int i = 0; i < 3; i++) {
        add_kernel<<<grid, block>>>((float*)da, (float*)db, (float*)dc, size);
        // single_kernel<<<grid, block>>>(da, db, size);
    }

    cudaMemcpy(hc, dc, size * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < size; i++) {
    //     assert(hc[i] == i * 2);
    // }

    return 0;
}