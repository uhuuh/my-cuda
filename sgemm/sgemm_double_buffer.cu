#include <cuda_runtime.h>
#include <cstdio>


template<int tile_size, int frag_size>
__global__
void sgemm_double_buffer_kernel(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int tile_row = tile_size;
    constexpr int thread_size = (tile_size / frag_size) * (tile_size / frag_size);
    constexpr int tile_col = thread_size / tile_size;
    constexpr int tile_read_num = tile_row * tile_col / thread_size;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * blockDim.x;
    auto block_a = A + bx * tile_size;
    auto block_b = B + by * tile_size * ldb;
    auto block_c = C + bx * tile_size + by * tile_size * ldc;

    auto gmem_tile_a = block_a;
    auto gmem_tile_b = block_b;
    __shared__ float smem_tiles_a[2][tile_row * tile_col];
    __shared__ float smem_tiles_b[2][tile_col * tile_row];
    float frag_acc[frag_size * frag_size] = {};

    int tile_x = tid % tile_row;
    int tile_y = tid / tile_row;
    auto smem_tile_a = smem_tiles_a[1];
    auto smem_tile_b = smem_tiles_b[1];
    for (int cnt = 0; cnt < tile_read_num; cnt += 1) {
        int tile_y2 = tile_y * tile_read_num + cnt;
        smem_tile_a[tile_x + tile_y2 * tile_row] = gmem_tile_a[tile_x + tile_y2 * lda];
        smem_tile_b[tile_y2 + tile_x * tile_col] = gmem_tile_b[tile_y2 + tile_x * ldb];
    }
    __syncthreads();

    gmem_tile_a += tile_col * lda;
    gmem_tile_b += tile_col;

    int tile_move_num = K / tile_col;
    for (int tile_move_cnt = 0; tile_move_cnt < tile_move_num; tile_move_cnt += 1) {
        if (tile_move_cnt < tile_move_num - 1) {
            smem_tile_a = smem_tiles_a[tile_move_cnt % 2];
            smem_tile_b = smem_tiles_b[tile_move_cnt % 2];
            for (int cnt = 0; cnt < tile_read_num; cnt += 1) {
                int tile_y2 = tile_y * tile_read_num + cnt;
                smem_tile_a[tile_x + tile_y2 * tile_row] = gmem_tile_a[tile_x + tile_y2 * lda];
                smem_tile_b[tile_y2 + tile_x * tile_col] = gmem_tile_b[tile_y2 + tile_x * ldb];
            }

            gmem_tile_a += tile_col * lda;
            gmem_tile_b += tile_col;
        }

        smem_tile_a = smem_tiles_a[(tile_move_cnt + 1) % 2];
        smem_tile_b = smem_tiles_b[(tile_move_cnt + 1) % 2];
        for (int tile_layer_offset = 0; tile_layer_offset < tile_col; tile_layer_offset += 1) {
            auto smem_frag_a = smem_tile_a + tx * frag_size + tile_layer_offset * tile_row;
            auto smem_frag_b = smem_tile_b + tile_layer_offset + ty * frag_size * tile_col;
            for (int frag_x = 0; frag_x < frag_size; frag_x += 1) {
                auto row_val = smem_frag_a[frag_x];
                for (int frag_y = 0; frag_y < frag_size; frag_y += 1) {
                    frag_acc[frag_x + frag_y * frag_size] += row_val * smem_frag_b[frag_y * tile_col];
                }
            }
        }
        __syncthreads();
    }

    auto gmem_frag_c = block_c + tx * frag_size + ty * frag_size * ldc;
    for (int frag_x = 0; frag_x < frag_size; frag_x += 1) {
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

    constexpr int tile_size = 64;
    constexpr int frag_size = 4;
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    TORCH_CHECK(M % tile_size == 0, "Incompatible matrix dimensions");
    TORCH_CHECK(N % tile_size == 0, "Incompatible matrix dimensions");
    TORCH_CHECK(M >= tile_size, "Incompatible matrix dimensions");
    TORCH_CHECK(N >= tile_size, "Incompatible matrix dimensions");
    TORCH_CHECK(K >= tile_size, "Incompatible matrix dimensions");

    // 创建输出张量
    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    dim3 block(tile_size / frag_size, tile_size / frag_size);
    dim3 grid((N + tile_size - 1) / tile_size, (M + tile_size - 1) / tile_size);
    sgemm_double_buffer_kernel<tile_size, frag_size><<<grid, block>>>(
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
    m.def("sgemm_double_buffer", &launch);
}