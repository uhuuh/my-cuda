#include <cuda_runtime.h>
#include <cstdio>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template<int tile_row_size, int tile_col_size, int frag_size>
__global__ __launch_bounds__(256)
void sgemm_memory_coalesce_kernel(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = tx + ty * blockDim.x;

    float* gmem_tile_a = A + bx * tile_row_size;
    float* gmem_tile_b = B + by * tile_row_size * ldb;
    __shared__ float smem_tiles_a[2][tile_row_size * tile_col_size];
    __shared__ float smem_tiles_b[2][tile_col_size * tile_row_size];
    float frag_acc[frag_size * frag_size] = {};

    const int tile_x1 = (tid % (tile_row_size / 4)) * 4;
    const int tile_y1 = tid / (tile_row_size / 4);
    const int tile_x2 = (tid % (tile_col_size / 4)) * 4;
    const int tile_y2 = tid / (tile_col_size / 4);
    const int gmem_tile_a_idx = tile_x1 + tile_y1 * lda;
    const int gmem_tile_b_idx = tile_x2 + tile_y2 * ldb;
    const int smem_tile_a_idx = tile_x1 + tile_y1 * tile_row_size;
    const int smem_tile_b_idx = tile_x2 * tile_row_size + tile_y2;
    float4 move_temp_row_val;
    float4 move_temp_col_val;

    FLOAT4(smem_tiles_a[1][smem_tile_a_idx]) = FLOAT4(gmem_tile_a[gmem_tile_a_idx]);
    move_temp_col_val = FLOAT4(gmem_tile_b[gmem_tile_b_idx]);
    smem_tiles_b[1][smem_tile_b_idx] = move_temp_col_val.x;
    smem_tiles_b[1][smem_tile_b_idx + tile_row_size] = move_temp_col_val.y;
    smem_tiles_b[1][smem_tile_b_idx + tile_row_size * 2] = move_temp_col_val.z;
    smem_tiles_b[1][smem_tile_b_idx + tile_row_size * 3] = move_temp_col_val.w;
    __syncthreads();

    const int tile_move_num = K / tile_col_size - 1;
    float frag_row_vals[frag_size];
    float frag_col_vals[frag_size];
#pragma unroll
    for (int tile_move_cnt = 0; tile_move_cnt < tile_move_num; tile_move_cnt += 1) {
      	const int write_smem_idx = tile_move_cnt % 2;
        const int read_smem_idx = (tile_move_cnt + 1) % 2;

        move_temp_row_val = FLOAT4((gmem_tile_a + tile_col_size * lda * (tile_move_cnt + 1))[gmem_tile_a_idx]);
        move_temp_col_val = FLOAT4((gmem_tile_b + tile_col_size * (tile_move_cnt + 1))[gmem_tile_b_idx]);

#pragma unroll
        for (int tile_layer_offset = 0; tile_layer_offset < tile_col_size; tile_layer_offset += 1) {
            float* smem_tile_a_ptr = smem_tiles_a[read_smem_idx] + tile_layer_offset * tile_row_size;
            float* smem_tile_b_ptr = smem_tiles_b[read_smem_idx] + tile_layer_offset * tile_row_size;
            FLOAT4(frag_row_vals[0]) = FLOAT4(smem_tile_a_ptr[tx * frag_size / 2]);
            FLOAT4(frag_col_vals[0]) = FLOAT4(smem_tile_b_ptr[ty * frag_size / 2]);
            FLOAT4(frag_row_vals[frag_size / 2]) = FLOAT4(smem_tile_a_ptr[tx * frag_size / 2 + tile_row_size / 2]);
            FLOAT4(frag_col_vals[frag_size / 2]) = FLOAT4(smem_tile_b_ptr[ty * frag_size / 2 + tile_row_size / 2]);

#pragma unroll
            for (int frag_x = 0; frag_x < frag_size; frag_x += 1) {
#pragma unroll
                for (int frag_y = 0; frag_y < frag_size; frag_y += 1) {
                    frag_acc[frag_x + frag_y * frag_size] += frag_row_vals[frag_x] * frag_col_vals[frag_y];
                }
            }
        }

        FLOAT4(smem_tiles_a[write_smem_idx][smem_tile_a_idx]) = move_temp_row_val;
        smem_tiles_b[write_smem_idx][smem_tile_b_idx] = move_temp_col_val.x;
        smem_tiles_b[write_smem_idx][smem_tile_b_idx + tile_row_size] = move_temp_col_val.y;
        smem_tiles_b[write_smem_idx][smem_tile_b_idx + tile_row_size * 2] = move_temp_col_val.z;
        smem_tiles_b[write_smem_idx][smem_tile_b_idx + tile_row_size * 3] = move_temp_col_val.w;
        __syncthreads();
    }

#pragma unroll
    for (int tile_layer_offset = 0; tile_layer_offset < tile_col_size; tile_layer_offset += 1) {
        float* smem_tile_a_ptr = smem_tiles_a[0] + tile_layer_offset * tile_row_size;
        float* smem_tile_b_ptr = smem_tiles_b[0] + tile_layer_offset * tile_row_size;
        FLOAT4(frag_row_vals[0]) = FLOAT4(smem_tile_a_ptr[tx * frag_size / 2]);
        FLOAT4(frag_col_vals[0]) = FLOAT4(smem_tile_b_ptr[ty * frag_size / 2]);
        FLOAT4(frag_row_vals[frag_size / 2]) = FLOAT4(smem_tile_a_ptr[tx * frag_size / 2 + tile_row_size / 2]);
        FLOAT4(frag_col_vals[frag_size / 2]) = FLOAT4(smem_tile_b_ptr[ty * frag_size / 2 + tile_row_size / 2]);

#pragma unroll
        for (int frag_x = 0; frag_x < frag_size; frag_x += 1) {
#pragma unroll
            for (int frag_y = 0; frag_y < frag_size; frag_y += 1) {
              	// TODO 这里没有用alpha和beta
                frag_acc[frag_x + frag_y * frag_size] += frag_row_vals[frag_x] * frag_col_vals[frag_y];
            }
        }
    }

    float* gemm_tile_c = C + bx * tile_row_size + by * tile_row_size * ldc +
        tx * frag_size / 2 + ty * frag_size / 2 * ldc;
#pragma unroll
    for (int i = 0; i < 2; i += 1) {
#pragma unroll
        for (int j = 0; j < 2; j += 1) {
#pragma unroll
            for (int k = 0; k < 4; k += 1) {
                FLOAT4(gemm_tile_c[i * tile_row_size / 2 + (j * tile_row_size / 2 + k) * ldc]) = FLOAT4(frag_acc[i * 4 + (j * 4 + k) * frag_size]);
            }
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