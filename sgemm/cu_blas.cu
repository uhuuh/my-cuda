#include <torch/extension.h>
#include <torch/extension.h>
#include <cublas_v2.h>


torch::Tensor my_gemm(torch::Tensor A, torch::Tensor B) {
    // 1. 检查
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");

    // 2. 创建输出
    auto C = torch::zeros(
        {A.size(0), B.size(1)},
        A.options()
    );

    // 3. 取 raw pointer
    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // 4. 拿 PyTorch 的 cublas handle
    auto handle = at::cuda::getDefaultCUDABlasHandle();

    float alpha = 1.0f, beta = 0.0f;

    // 5. 调用 cuBLAS
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        A.size(0), B.size(1), A.size(1),
        &alpha,
        A_ptr, A.size(0),
        B_ptr, A.size(1),
        &beta,
        C_ptr, A.size(0)
    );

    return C;
}

TORCH_LIBRARY(my_ops, m) {
    m.def("gemm(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
    m.impl("gemm", my_gemm);
}
