#include <cuda_runtime.h>
#include <cute/tensor.hpp>


template<int ts>
__device__
void sgemm_smem_cache(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    __shared__ float tile_a[ts * ts];
    __shared__ float tile_b[ts * ts];
    float res = 0;

    for (int ti = 0; ti < K; ti += ts) {
    }
}


