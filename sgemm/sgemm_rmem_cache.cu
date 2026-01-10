#include <cuda_runtime.h>


template<int tile_size, int frag_size>
__global__
void sgemm_rmem_cache_kernel(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    A = A + bx * tile_size;
    B = B + by * tile_size * ldb;
    C = C + bx * tile_size + by * tile_size * ldc;

    __shared__ float tile_a[tile_size * tile_size];
    __shared__ float tile_b[tile_size * tile_size];
    float acc = 0;
    for (int ti = 0; ti * tile_size < K; ti += 1) {
        tile_a[tx + ty * tile_size] = A[tx + ty * lda];
        tile_b[tx + ty * tile_size] = B[tx + ty * ldb];
        __syncthreads();

        A += tile_size * lda;
        B += tile_size;

        for (int k = 0; k < tile_size; ++k) {
            acc += tile_a[tx + k * tile_size] * tile_b[k + ty * tile_size];
        }
    }
    C[tx + ty * ldc] = alpha * acc + beta * C[tx + ty * ldc];
}


#include <torch/extension.h>
torch::Tensor launch(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix dimensions");

    constexpr int tile_size = 64;
    constexpr int frag_size = 4;
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    TORCH_CHECK(M % tile_size == 0, "Incompatible matrix dimensions");
    TORCH_CHECK(N % tile_size == 0, "Incompatible matrix dimensions");

    // 创建输出张量
    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dim3 grid((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);
    dim3 block(tile_size / frag_size, tile_size);
    sgemm_rmem_cache_kernel<tile_size, frag_size><<<grid, block>>>(
        B.data_ptr<float>(),
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        alpha, beta,
        N, M, K,
        N, K, N
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(cudaGetLastError()));

    return C;
}

// 使用TORCH_LIBRARY注册算子
TORCH_LIBRARY(my, m) {
    m.def("sgemm_rmem_cache(Tensor A, Tensor B) -> Tensor");
}

// 为CUDA设备注册实现
TORCH_LIBRARY_IMPL(my, CUDA, m) {
    m.impl("sgemm_rmem_cache", launch);
}