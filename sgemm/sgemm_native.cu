#include <cuda.h>


__device__
void sgemm_native(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    float acc = 0;
    for (int k = 0; k < K; ++k) {
        acc += A[tx + k * lda] * B[k + ty * ldb];
    }
    C[tx + ty * ldc] = alpha * acc + beta * C[tx + ty * ldc];
}
