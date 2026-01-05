#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
using namespace std;

#define cuda_check(call) do{\
  cudaError_t cuda_ret = (call);\
  if(cuda_ret != cudaSuccess){\
    printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
    printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
    printf("  In the function call %s\n", #call);\
    exit(1);\
  }\
}while(0)

#define i2(mat, col, i, j) mat[(i) * (col) + (j)]
#define i4(mat, x, y, a, b) mat[(x) * (a) + (y) * (b)]

#define itx threadIdx.x
#define ity threadIdx.y
#define ibx blockIdx.x
#define iby blockIdx.y
#define dgx gridDim.x
#define dgy gridDim.y
#define ddx blockDim.x
#define ddy blockDim.y

#define vload(v1,addr) v1 = *((float4 *)(addr))
#define vstore(addr,v1) *((float4 *)(addr)) = v1
#define CONCAT(a, b) a##b

template <typename T, int m, int n>
__host__ __device__
void mat_print(T* data) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", data[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T, int row_step, int col_step>
struct Mat {
    T* data;
    __host__ __device__ Mat(T& data): data(&data) {}
    __host__ __device__ Mat(T* data): data(data) {}
    __host__ __device__ void set_data(T* data) {
        this->data = data;
    }
    __host__ __device__ void set_data(T& data) {
        this->data = &data;
    }
    __host__ __device__ T& index(int i, int j) {
        return data[i * row_step + j * col_step];
    }
};

template <typename T, int stride>
struct Matv1 {
    T* data;
    __host__ __device__ Matv1(T& data): data(&data) {}
    __host__ __device__ Matv1(T* data): data(data) {}
    template <int step=1>
    __host__ __device__ T& index(int i, int j) {
        return data[i * stride * step + j * step];
    }
    template <int step=1>
    __host__ __device__ auto slice(int i, int j) {
        return Matv1<T, stride>(data + i * stride * step + j * step);
    }
};

template <typename T, int m, int n>
struct MatManager {
    T* host_ptr = nullptr;
    T* device_ptr = nullptr;
    MatManager(bool is_random = true) {
        host_ptr = new T[m * n];
        if (is_random) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    //Mat<T, n, 1>(host_ptr).index(i, j) = rand() / RAND_MAX;
                    Mat<T, n, 1>(host_ptr).index(i, j) = i * n + j;
                }
            }
        }

        cuda_check(cudaMalloc(&device_ptr, sizeof(T) * m * n));
    }
    void move(bool to_device) {
        if (to_device) {
            cuda_check(cudaMemcpy(device_ptr, host_ptr, sizeof(T) * m * n, cudaMemcpyHostToDevice));
        } else {
            cuda_check(cudaMemcpy(host_ptr, device_ptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost));
        }
    }
    void print() {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                printf("%f ", Mat<T, n, 1>(host_ptr).index(i, j));
            }
            printf("\n");
        }
    }
    ~MatManager() {
        if (host_ptr) {
            delete [] host_ptr;
        }
        if (device_ptr) {
            cuda_check(cudaFree(device_ptr));
        }
    }
};

// 每个线程块计算结果矩阵中的一块内容
template <typename T, int block_size, int m, int n, int k>
__global__
void matmul_v0(T* a_ptr, T* b_ptr, T* c_ptr) {
    Mat<T, block_size * k, 0> a_block(a_ptr);
    Mat<T, 0, block_size> b_block(b_ptr);
    Mat<T, block_size * k, block_size> c_block(c_ptr);

    Mat<T, k, 1> a_tile(a_block.index(blockIdx.x, 0));
    Mat<T, n, 1> b_tile(b_block.index(0, blockIdx.y));
    Mat<T, k, 1> c_tile(c_block.index(blockIdx.x, blockIdx.y));

    T a = 0;
    for (int z = 0; z < k; ++z) {
        a += a_tile.index(threadIdx.x, z) * b_tile.index(z, threadIdx.y);
    }
    c_tile.index(threadIdx.x, threadIdx.y) = a;
    // if (blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("%d %d: %f\n", threadIdx.x, threadIdx.y, a);
    // }
}

// 线程块维护一个共享内容块
template <typename T, int block_size, int m, int n, int k>
__global__
void matmul_v1(T* a_ptr, T* b_ptr, T* c_ptr) {
    __shared__ T a_shared[block_size][block_size];
    __shared__ T b_shared[block_size][block_size];

    Mat<T, block_size * k, block_size> a_block(a_ptr);
    Mat<T, block_size * n, block_size> b_block(b_ptr);
    Mat<T, block_size * k, block_size> c_block(c_ptr);

    Mat<T, k, 1> c_tile(c_block.index(blockIdx.x, blockIdx.y));

    T result = 0;
    for (int block_z = 0; block_z < k / block_size; ++block_z) {
        Mat<T, k, 1> a_tile(a_block.index(blockIdx.x, block_z));
        Mat<T, n, 1> b_tile(b_block.index(block_z, blockIdx.y));
        a_shared[threadIdx.x][threadIdx.y] = a_tile.index(threadIdx.x, threadIdx.y);
        b_shared[threadIdx.x][threadIdx.y] = b_tile.index(threadIdx.x, threadIdx.y);
        __syncthreads();

        for (int tile_z = 0; tile_z < block_size; ++tile_z) {
            result += a_shared[threadIdx.x][tile_z] * b_shared[tile_z][threadIdx.y];
        }
        __syncthreads();
    }
    c_tile.index(threadIdx.x, threadIdx.y) = result;
}

// 每个线程处理更多元素
template <typename T, int block_size, int m, int n, int k, int tile_size=2>
__global__
void matmul_v2(T* a_ptr, T* b_ptr, T* c_ptr) {
    Matv1<T, k> a_mat(a_ptr);
    Matv1<T, n> b_mat(b_ptr);
    Matv1<T, k> c_mat(c_ptr);
    auto c_tile = c_mat.template slice<block_size>(blockIdx.x, blockIdx.y);

    __shared__ T a_shared[block_size][block_size];
    __shared__ T b_shared[block_size][block_size];
    Matv1<T, block_size> a_shared_block((T*)a_shared);
    Matv1<T, block_size> b_shared_block((T*)b_shared);
    auto a_shared_tile = a_shared_block.template slice<tile_size>(threadIdx.x, threadIdx.y);
    auto b_shared_tile = b_shared_block.template slice<tile_size>(threadIdx.x, threadIdx.y);

    T result[tile_size][tile_size] = {};
    for (int block_z = 0; block_z < k / block_size; ++block_z) {
        auto a_tile = a_mat.template slice<block_size>(blockIdx.x, block_z).template slice<tile_size>(threadIdx.x, threadIdx.y);
        auto b_tile = b_mat.template slice<block_size>(block_z, blockIdx.y).template slice<tile_size>(threadIdx.x, threadIdx.y);
        for (int inner_x = 0; inner_x < tile_size; ++inner_x) {
            for (int inner_y = 0; inner_y < tile_size; ++inner_y) {
                a_shared_tile.index(inner_x, inner_y) = a_tile.index(inner_x, inner_y);
                b_shared_tile.index(inner_x, inner_y) = b_tile.index(inner_x, inner_y);
            }
        }
        __syncthreads();

        for (int inner_x = 0; inner_x < tile_size; ++inner_x) {
            for (int inner_y = 0; inner_y < tile_size; ++inner_y) {
                for (int tile_z = 0; tile_z < block_size; ++tile_z) {
                    auto tile_x = threadIdx.x * tile_size + inner_x;
                    auto tile_y = threadIdx.y * tile_size + inner_y;
                    result[inner_x][inner_y] += a_shared[tile_x][tile_z] * b_shared[tile_z][tile_y];
                }
            }
        }
        __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        //     printf("------>\n");
        //     mat_print<T, block_size, block_size>((T*)a_shared);
        //     mat_print<T, block_size, block_size>((T*)b_shared);
        //     mat_print<T, tile_size, tile_size>((T*)result);
        // }
        // __syncthreads();
    }
    for (int inner_x = 0; inner_x < tile_size; ++inner_x) {
        for (int inner_y = 0; inner_y < tile_size; ++inner_y) {
            c_tile.index(inner_x, inner_y) = result[inner_x][inner_y];
        }
    }
}

// 寄存器分组
template <typename T, int block_size, int m, int n, int k, int tile_size=2, int reg_size=2>
__global__
void matmul_v3(T* a_ptr, T* b_ptr, T* c_ptr) {
    Matv1<T, k> a_mat(a_ptr);
    Matv1<T, n> b_mat(b_ptr);
    Matv1<T, k> c_mat(c_ptr);
    auto c_tile = c_mat.template slice<block_size>(blockIdx.x, blockIdx.y);

    __shared__ T a_shared[block_size][block_size];
    __shared__ T b_shared[block_size][block_size];
    Matv1<T, block_size> a_shared_block((T*)a_shared);
    Matv1<T, block_size> b_shared_block((T*)b_shared);
    auto a_shared_tile = a_shared_block.template slice<tile_size>(threadIdx.x, threadIdx.y);
    auto b_shared_tile = b_shared_block.template slice<tile_size>(threadIdx.x, threadIdx.y);

    T result[tile_size][tile_size] = {};
    for (int block_z = 0; block_z < k / block_size; ++block_z) {
        auto a_tile = a_mat.template slice<block_size>(blockIdx.x, block_z).template slice<tile_size>(threadIdx.x, threadIdx.y);
        auto b_tile = b_mat.template slice<block_size>(block_z, blockIdx.y).template slice<tile_size>(threadIdx.x, threadIdx.y);
        for (int inner_x = 0; inner_x < tile_size; ++inner_x) {
            for (int inner_y = 0; inner_y < tile_size; ++inner_y) {
                a_shared_tile.index(inner_x, inner_y) = a_tile.index(inner_x, inner_y);
                b_shared_tile.index(inner_x, inner_y) = b_tile.index(inner_x, inner_y);
            }
        }
        __syncthreads();

        for (int inner_x = 0; inner_x < tile_size; ++inner_x) {
            for (int inner_y = 0; inner_y < tile_size; ++inner_y) {
                for (int tile_z = 0; tile_z < block_size; ++tile_z) {
                    auto tile_x = threadIdx.x * tile_size + inner_x;
                    auto tile_y = threadIdx.y * tile_size + inner_y;
                    result[inner_x][inner_y] += a_shared[tile_x][tile_z] * b_shared[tile_z][tile_y];
                }
            }
        }
        __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        //     printf("------>\n");
        //     mat_print<T, block_size, block_size>((T*)a_shared);
        //     mat_print<T, block_size, block_size>((T*)b_shared);
        //     mat_print<T, tile_size, tile_size>((T*)result);
        // }
        // __syncthreads();
    }
    for (int inner_x = 0; inner_x < tile_size; ++inner_x) {
        for (int inner_y = 0; inner_y < tile_size; ++inner_y) {
            c_tile.index(inner_x, inner_y) = result[inner_x][inner_y];
        }
    }
}

struct Timer {
    cudaEvent_t begin, end;
    Timer() {
        cuda_check(cudaEventCreate(&begin));
        cuda_check(cudaEventCreate(&end));
        cudaEventRecord(begin);
    }
    void start() {
        cudaDeviceSynchronize();
        cudaEventRecord(end);
        cudaEventSynchronize(begin);
        cudaEventSynchronize(end);
    }
    float stop() {
        float time = 0;
        cudaEventElapsedTime(&time, begin, end);
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        return time;
    }
};

void test_acc(int case_id) {
    const int block_size = 2;
    const int tile_size = 2;
    const int m = 4;
    const int n = 4;
    const int k = 4;

    MatManager<float, m, k> A(true);
    MatManager<float, k, n> B(true);
    MatManager<float, m, n> C(true);
    A.move(true);
    B.move(true);

    Timer t;
    t.start();

    if (case_id == 0) {
        matmul_v0<float, block_size, m, n, k>
            <<<dim3(m / block_size, n / block_size), dim3(block_size, block_size)>>>
            ((float*)A.device_ptr, (float*)B.device_ptr, (float*)C.device_ptr);
    } else if (case_id == 1) {
        matmul_v1<float, block_size, m, n, k>
            <<<dim3(m / block_size, n / block_size), dim3(block_size, block_size)>>>
            ((float*)A.device_ptr, (float*)B.device_ptr, (float*)C.device_ptr);
    } else if (case_id == 2) {
        matmul_v2<float, block_size, m, n, k>
            <<<dim3(m / block_size, n / block_size), dim3(block_size / tile_size, block_size / tile_size)>>>
            ((float*)A.device_ptr, (float*)B.device_ptr, (float*)C.device_ptr);
    } else {
        exit(1);
    }

    t.stop();

    C.move(false);
    // C.print();

    bool is_ok = true;
    float res[][n] = {
         {56,  62,  68,  74},
         {152, 174, 196, 218},
         {248, 286, 324, 362},
         {344, 398, 452, 506}
    };
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (C.host_ptr[i * n + j] != res[i][j]) {
                is_ok = false;
                printf("error, output[%d][%d] = %f, res[%d][%d] = %f\n", i, j, C.host_ptr[i * n + j], i, j, res[i][j]);
            }
        }
    }
    printf("test %d acc: %s\n", case_id, is_ok ? "pass" : "fail");
}

void test_perf(int case_id) {
    const int block_size = 32;
    const int tile_size = 4;
    const int m = 32 * 128;
    const int n = 32 * 128;
    const int k = 32 * 128;
    const int iter = 10;
    float per_results[iter];

    MatManager<float, m, k> A(true);
    MatManager<float, k, n> B(true);
    MatManager<float, m, n> C(true);
    A.move(true);
    B.move(true);

    for (int i = 0; i < iter; ++i) {
        Timer t;
        t.start();

        if (case_id == 0) {
            matmul_v0<float, block_size, m, n, k>
                <<<dim3(m / block_size, n / block_size), dim3(block_size, block_size)>>>
                ((float*)A.device_ptr, (float*)B.device_ptr, (float*)C.device_ptr);
        } else if (case_id == 1) {
            matmul_v1<float, block_size, m, n, k>
                <<<dim3(m / block_size, n / block_size), dim3(block_size, block_size)>>>
                ((float*)A.device_ptr, (float*)B.device_ptr, (float*)C.device_ptr);
        } else if (case_id == 2) {
            matmul_v2<float, block_size, m, n, k>
                <<<dim3(m / block_size, n / block_size), dim3(block_size / tile_size, block_size / tile_size)>>>
                ((float*)A.device_ptr, (float*)B.device_ptr, (float*)C.device_ptr);
        } else {
            exit(1);
        }

        auto time = t.stop();
        per_results[i] = time;
    }

    std::string times;
    for (int i = 0; i < iter; ++i) {
        times += std::to_string(per_results[i]);
        times += " ";
    }
    printf("test %d perf: time=%sms\n", case_id, times.c_str());
}



int main() {
    test_acc(0);
    test_acc(1);
    test_acc(2);
    test_perf(0);
    test_perf(1);
    test_perf(2);
    return 0;
}