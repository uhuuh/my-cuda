#include <cuda_runtime.h>


template<int ts>
__global__
void sgemm_smem_cache_kernel(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    A = A + bx * ts;
    B = B + by * ts * ldb;
    C = C + bx * ts + by * ts * ldc;

    __shared__ float tile_a[ts * ts];
    __shared__ float tile_b[ts * ts];
    float acc = 0;
    for (int ti = 0; ti * ts < K; ti += 1) {
        tile_a[tx + ty * ts] = A[tx + ty * lda];
        tile_b[tx + ty * ts] = B[tx + ty * ldb];
        __syncthreads();

        A += ts * lda;
        B += ts;

        for (int k = 0; k < ts; ++k) {
            acc += tile_a[tx + k * ts] * tile_b[k + ty * ts];
        }
    }
    C[bx + by * ldc] = alpha * acc + beta * C[bx + by * ldc];
}


#include <torch/extension.h>
torch::Tensor launch(torch::Tensor A, torch::Tensor B) {
    // 输入验证
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix dimensions");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // 创建输出张量
    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    // 常量参数
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int lda = M;
    const int ldb = K;
    const int ldc = M;

    // 分块大小 (可根据硬件调整)
    constexpr int ts = 16;

    // 计算grid和block尺寸
    dim3 block(ts, ts);
    dim3 grid((M + ts - 1) / ts, (N + ts - 1) / ts);

    // 启动kernel
    sgemm_smem_cache_kernel<ts><<<grid, block>>>(
        B.data_ptr<float>(),
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        alpha, beta,
        M, N, K,
        ldb, lda, ldc
    );

    // 检查CUDA错误
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(cudaGetLastError()));

    return C;
}

// 使用TORCH_LIBRARY注册算子
TORCH_LIBRARY(my, m) {
    m.def("sgemm_smem_cache(Tensor A, Tensor B) -> Tensor");
}

// 为CUDA设备注册实现
TORCH_LIBRARY_IMPL(my, CUDA, m) {
    m.impl("sgemm_smem_cache", launch);
}