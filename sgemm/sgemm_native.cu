#include <cuda_runtime.h>


__global__
void sgemm_native_kernel(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    float acc = 0;
    for (int k = 0; k < K; ++k) {
        acc += A[tx + k * lda] * B[k + ty * ldb];
    }
    C[tx + ty * ldc] = alpha * acc + beta * C[tx + ty * ldc];
}


#include <torch/extension.h>
torch::Tensor launch(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix dimensions");

    constexpr int tile_size = 16;
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

    dim3 block(tile_size, tile_size);
    dim3 grid((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);
    sgemm_native_kernel<<<grid, block>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_native", &launch);
}