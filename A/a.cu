#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
using namespace std;

#define cuda_check(call) do{\
  cudaError_t cuda_ret = (call);\
  if(cuda_ret != cudaSuccess){\
    printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
    printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
    printf("  In the function call %s\n", #call);\
    exit(1);\
  }\
}while(0)

#define i2(mat, col, i, j) mat[(i) * (col) + (j)]

const bool debug = true;

struct Mat {
    float* h = nullptr;
    float* d = nullptr;
    int m = 0, n = 0;
    Mat(int m, int n): m(m), n(n) {
        h = (float*)malloc(get_size());
        cuda_check(cudaMalloc((void**)&d, get_size()));
    }
    void init() {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                i2(h, n, i, j) = (float)rand() / RAND_MAX;
                // i2(h, n, i, j) = 1;
            }
        }
        if (debug) {
            print();
        }
    }
    void print() {
        printf("-------------------> mat %p [%d %d]\n", this, m, n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                printf("%f ", index(i, j, true));
            }
            printf("\n");
        }
        printf("\n");
    }
    int get_size() {
        return m * n * sizeof(float);
    }
    inline float index(int i, int j, bool is_host) {
        if (is_host) {
            return i2(h, n, i, j);
        } else {
            return i2(d, n, i, j);
        }
    }
    void move(bool to_device) {
        if (to_device) {
            cuda_check(cudaMemcpy(d, h, get_size(), cudaMemcpyHostToDevice));
        } else {
            cuda_check(cudaMemcpy(h, d, get_size(), cudaMemcpyDeviceToHost));
        }
        cudaDeviceSynchronize();
    }
    ~Mat() {
        free(h);
        cuda_check(cudaFree(d));
    }
};

__global__
void matmul(int m, int n, int k, float* A, float* B, float* C) {
    //printf("[%d %d] [%d %d]\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    const int tile_size = 32;
    A = A + blockIdx.x * tile_size * n + blockIdx.y * tile_size; 
    B = B + blockIdx.x * tile_size * n + blockIdx.y * tile_size; 
    C = C + blockIdx.x * tile_size * n + blockIdx.y * tile_size; 

    float a = 0;
    for (int k = 0; k < tile_size; ++k) {
        a += i2(A, k, threadIdx.x, k) * i2(B, n, k, threadIdx.y);
    }
    i2(C, m, threadIdx.x, threadIdx.y) = a;
    for (int k = 0; k < tile_size; ++k) {
        a += i2(A, k, threadIdx.x, k) * i2(B, n, k, threadIdx.y);
    }
    for (int k = 0; k < tile_size; ++k) {
        a += i2(A, k, threadIdx.x, k) * i2(B, n, k, threadIdx.y);
    }
    printf("%d %d -> %f\n", threadIdx.x, threadIdx.y, a);
}

void check(Mat* A, Mat* B, Mat* C) {
    int m = A->m, n = B->n, k = A->n;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float a = 0;
            for (int ij = 0; ij < k; ++ij) {
                a += i2(A->h, A->n, i, ij) * i2(B->h, B->n, ij, j);
                // printf("------> %f\n", i2(A->h, A->n, i, ij) * i2(B->h, B->n, ij, j));
            }
            // if (a != i2(C->h, C->n, i, j)) {
            float diff = a - i2(C->h, C->n, i, j);
            if (abs(diff) > 0.001) {
                // A->print();
                // B->print();
                // C->print();
                printf("in [%d][%d], %f is fail, %f is ok, %f is diff\n", i, j, i2(C->h, C->n, i, j), a, diff);
                printf("A has ");
                for (int ij = 0; ij < k; ++ij) printf("%f ", A->index(i, ij, true));
                printf("\n");
                printf("B has ");
                for (int ij = 0; ij < k; ++ij) printf("%f ", B->index(ij, j, true));
                printf("\n");
                exit(0);
            }
        }
    }
}

const int m = 32;
const int n = 32;
const int k = 32;

int main() {
    srand(42);

    Mat A(m, k);
    Mat B(k, n);
    Mat C(m, n);
    A.init();
    B.init();
    A.move(true); 
    B.move(true);

    matmul<<<dim3(m / 32, n / 32), dim3(32, 32)>>>(m, n, k, (float*)A.d, (float*)B.d, (float*)C.d);
    cudaDeviceSynchronize();

    C.move(false);
    check(&A, &B, &C);
    return 0;
}