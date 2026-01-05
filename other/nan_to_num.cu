// my_ops_cuda.cu
#include <torch/extension.h>
#include <cuda_runtime.h>


__global__ void my_add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

torch::Tensor my_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    my_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );
    return out;
}

__global__ void my_nan_to_num_kernel(const float* inp, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    auto a = inp[i];
    if (isnan(a)) {
        out[i] = 0.0f;
    } else if (isinf(a)) {
        out[i] = 0.0f;
    } else {
        out[i] = a;
    }
}

torch::Tensor my_nan_to_num_cuda(torch::Tensor inp) {
    auto out = torch::empty_like(inp);
    int n = inp.numel();
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    my_nan_to_num_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );
    return out;
}

__global__ void my_nan_to_num_kernel_v2(const float* inp, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 a = *(((float4*)inp) + i);
    float4 b;

    if (isnan(a.x)) {
        b.x = 0.0f;
    } else if (isinf(a.x)) {
        b.x = 0.0f;
    } else {
        b.x = a.x;
    }

    if (isnan(a.y)) {
        b.y = 0.0f;
    } else if (isinf(a.y)) {
        b.y = 0.0f;
    } else {
        b.y = a.y;
    }

    if (isnan(a.z)) {
        b.z = 0.0f;
    } else if (isinf(a.z)) {
        b.z = 0.0f;
    } else {
        b.z = a.z;
    }

    if (isnan(a.w)) {
        b.w = 0.0f;
    } else if (isinf(a.w)) {
        b.w = 0.0f;
    } else {
        b.w = a.w;
    }
    *((float4*)out + i) = b;
}

torch::Tensor my_nan_to_num_cuda_v2(torch::Tensor inp) {
    int n = inp.numel();
    int aligned_n = ((n + 3) / 4) * 4;  // 向上 pad 到 4 的倍数
    auto tmp_out = torch::empty({aligned_n}, inp.options());

    int threads = 128;
    int blocks = (aligned_n + threads - 1) / threads;
    my_nan_to_num_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        tmp_out.data_ptr<float>(),
        n
    );

    auto out = tmp_out.narrow(0, 0, n).view(inp.sizes());

    return out;
}

__global__ void test_mem(const float* inp, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    out[i] = inp[i];
}

torch::Tensor test_mem_cuda(torch::Tensor inp) {
    auto out = torch::empty_like(inp);
    int n = inp.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    test_mem<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );
    return out;
}


TORCH_LIBRARY(my, m) {
    m.def("nan_to_num(Tensor x) -> Tensor");
    m.def("nan_to_num_v2(Tensor x) -> Tensor");
    m.def("add(Tensor x, Tensor y) -> Tensor");
    m.def("test_mem(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(my, CUDA, m) {
    m.impl("add", my_add_cuda);
    m.impl("nan_to_num", my_nan_to_num_cuda);
    m.impl("nan_to_num_v2", my_nan_to_num_cuda_v2);
    m.impl("test_mem", test_mem_cuda);
}
