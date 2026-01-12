#include <cuda_runtime.h>
#include <cstdio>


template<int tile_row_size, int tile_col_size, int frag_size>
__global__
void sgemm_memory_coalesce_kernel(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * blockDim.x;
    auto block_a = A + bx * tile_row_size;
    auto block_b = B + by * tile_row_size * ldb;
    auto block_c = C + bx * tile_row_size + by * tile_row_size * ldc;

    auto gmem_tile_a = block_a;
    auto gmem_tile_b = block_b;
    __shared__ float smem_tiles_a[2][tile_row_size * tile_col_size];
    __shared__ float smem_tiles_b[2][tile_col_size * tile_row_size];
    float frag_acc[frag_size * frag_size] = {};

    int tile_f4_row_size = tile_row_size / 4;
    int tile_f4_col_size = tile_col_size / 4;
    int tile_f4_x1 = tid % tile_f4_row_size;
    int tile_f4_y1 = tid / tile_f4_row_size;
    int tile_f4_x2 = tid % tile_f4_col_size;
    int tile_f4_y2 = tid / tile_f4_col_size;
    float4* temp_smem_a_ptrs[2] = {
        reinterpret_cast<float4*>(smem_tiles_a[0]) + tile_f4_x1 + tile_f4_y1 * tile_f4_row_size,
        reinterpret_cast<float4*>(smem_tiles_a[1]) + tile_f4_x1 + tile_f4_y1 * tile_f4_row_size
    };
    float4* temp_smem_b_ptrs[2] = {
        reinterpret_cast<float4*>(smem_tiles_b[0]) + tile_f4_y1 * tile_f4_row_size + tile_f4_x1,
        reinterpret_cast<float4*>(smem_tiles_b[1]) + tile_f4_y1 * tile_f4_row_size + tile_f4_x1
    };

    auto temp_smem_a = temp_smem_a_ptrs[1];
    auto temp_smem_b = temp_smem_b_ptrs[1];
    *temp_smem_a = reinterpret_cast<float4*>(gmem_tile_a)[tile_f4_x1 + tile_f4_y1 * (lda / 4)];
    auto tmep_gmem_b = reinterpret_cast<float4*>(gmem_tile_b) + tile_f4_x2 + tile_f4_y2 * (ldb / 4);
    temp_smem_b->x = tmep_gmem_b->x;
    temp_smem_b->y = tmep_gmem_b->y;
    temp_smem_b->z = tmep_gmem_b->z;
    temp_smem_b->w = tmep_gmem_b->w;
    __syncthreads();

    gmem_tile_a += tile_col_size * lda;
    gmem_tile_b += tile_col_size;

    float* smem_tile_a;
    float* smem_tile_b;
    int tile_move_num = K / tile_col_size;
    #pragma unroll
    for (int tile_move_cnt = 0; tile_move_cnt < tile_move_num; tile_move_cnt += 1) {
        if (tile_move_cnt < tile_move_num - 1) {
            temp_smem_a = temp_smem_a_ptrs[tile_move_cnt % 2];
            temp_smem_b = temp_smem_b_ptrs[tile_move_cnt % 2];

            *temp_smem_a = reinterpret_cast<float4*>(gmem_tile_a)[tile_f4_x1 + tile_f4_y1 * (lda / 4)];
            tmep_gmem_b = reinterpret_cast<float4*>(gmem_tile_b) + tile_f4_x2 + tile_f4_y2 * (ldb / 4);
            temp_smem_b->x = tmep_gmem_b->x;
            temp_smem_b->y = tmep_gmem_b->y;
            temp_smem_b->z = tmep_gmem_b->z;
            temp_smem_b->w = tmep_gmem_b->w;

            gmem_tile_a += tile_col_size * lda;
            gmem_tile_b += tile_col_size;
        }

        smem_tile_a = smem_tiles_a[(tile_move_cnt + 1) % 2];
        smem_tile_b = smem_tiles_b[(tile_move_cnt + 1) % 2];
        #pragma unroll
        for (int tile_layer_offset = 0; tile_layer_offset < tile_col_size; tile_layer_offset += 1) {
            auto smem_frag_a = smem_tile_a + tx * frag_size + tile_layer_offset * tile_row_size;
            auto smem_frag_b = smem_tile_b + tile_layer_offset + ty * frag_size * tile_col_size;
            #pragma unroll
            for (int frag_x = 0; frag_x < frag_size; frag_x += 1) {
                auto row_val = smem_frag_a[frag_x];
                #pragma unroll
                for (int frag_y = 0; frag_y < frag_size; frag_y += 1) {
                    frag_acc[frag_x + frag_y * frag_size] += row_val * smem_frag_b[frag_y * tile_col_size];
                }
            }
        }
        __syncthreads();
    }

    auto gmem_frag_c = block_c + tx * frag_size + ty * frag_size * ldc;
    #pragma unroll
    for (int frag_x = 0; frag_x < frag_size; frag_x += 1) {
        #pragma unroll
        for (int frag_y = 0; frag_y < frag_size; frag_y += 1) {
            gmem_frag_c[frag_x + frag_y * ldc] = alpha * frag_acc[frag_x + frag_y * frag_size] + beta * gmem_frag_c[frag_x + frag_y * ldc];
        }
    }
}


#include <torch/extension.h>
torch::Tensor launch(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix dimensions");

    constexpr int tile_row_size = 128;
    constexpr int tile_col_size = 8;
    constexpr int frag_size = 8;
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    TORCH_CHECK(M % tile_row_size == 0, "Incompatible matrix dimensions");
    TORCH_CHECK(N % tile_row_size == 0, "Incompatible matrix dimensions");
    TORCH_CHECK(M >= tile_row_size, "Incompatible matrix dimensions");
    TORCH_CHECK(N >= tile_row_size, "Incompatible matrix dimensions");
    TORCH_CHECK(K >= tile_row_size, "Incompatible matrix dimensions");

    // 创建输出张量
    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dim3 block(tile_row_size / frag_size, tile_row_size / frag_size);
    dim3 grid((N + tile_row_size - 1) / tile_row_size, (M + tile_row_size - 1) / tile_row_size);
    sgemm_memory_coalesce_kernel<tile_row_size, tile_col_size, frag_size><<<grid, block>>>(
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
    m.def("sgemm_memory_coalesce(Tensor A, Tensor B) -> Tensor");
}

// 为CUDA设备注册实现
TORCH_LIBRARY_IMPL(my, CUDA, m) {
    m.impl("sgemm_memory_coalesce", launch);
}