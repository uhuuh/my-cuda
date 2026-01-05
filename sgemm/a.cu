#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
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

const bool debug = true;



template <int m, int n>
struct Mat {
    float* h = nullptr;
    float* d = nullptr;
    string name;
    Mat(const string &name): name(name) {
        h = (float*)malloc(get_size());
        cuda_check(cudaMalloc((void**)&d, get_size()));
    }
    void init() {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                i2(h, n, i, j) = (float)rand() / RAND_MAX;
                // i2(h, n, i, j) = 1;
            }
        }
        if (debug) {
            print();
        }
    }
    void print() {
        return;

        int saved_stdout = dup(STDOUT_FILENO); 
        freopen(name.c_str(), "w", stdout);

        printf("-------------------> mat %p [%d %d]\n", this, m, n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                printf("%f ", index(i, j, true));
            }
            printf("\n");
        }
        printf("\n");

        dup2(saved_stdout, STDOUT_FILENO);  // 还原 stdout
        close(saved_stdout);                // 关闭备份的描述符
    }
    int get_size() {
        return m * n * sizeof(float);
    }
    inline float index(int i, int j, bool is_host) {
        if (is_host) {
            return i2(h, n, i, j);
        } else {
            return i2(d, n, i, j);
        }
    }
    void move(bool to_device) {
        if (to_device) {
            cuda_check(cudaMemcpy(d, h, get_size(), cudaMemcpyHostToDevice));
        } else {
            cuda_check(cudaMemcpy(h, d, get_size(), cudaMemcpyDeviceToHost));
        }
        cudaDeviceSynchronize();
    }
    ~Mat() {
        free(h);
        cuda_check(cudaFree(d));
    }
};

__global__
void matmul(int m, int n, int k, float* A, float* B, float* C) {
    //printf("[%d %d] [%d %d]\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    const int tile_size = 32;
    A = A + blockIdx.x * tile_size * k;
    B = B + blockIdx.y * tile_size; 
    C = C + blockIdx.x * tile_size * n + blockIdx.y * tile_size; 

    float a = 0;
    for (int z = 0; z < k; ++z) {
        a += i2(A, k, threadIdx.x, z) * i2(B, n, z, threadIdx.y);
        // if (debug && ibx == 0 && iby == 1 && itx == 0 && ity == 0) {
        //     printf("A %d %d %f\n", threadIdx.x, z, i2(A, k, threadIdx.x, z));
        //     printf("B %d %d %f\n", z, threadIdx.y, i2(B, n, z, threadIdx.y));
        //     printf("%f\n", a);
        // }
    }
    i2(C, n, threadIdx.x, threadIdx.y) = a;
    // printf("%d %d -> %f\n", threadIdx.x, threadIdx.y, a);
}

// 使用共享内存
template <int tile = 32>
__global__
void matmul_v2(float* A, float* B, float* C, int m, int n, int k) {
     float* A_chip = NULL;
     float* B_chip = NULL;
     float* C_chip = NULL;
     __shared__ float A_buf[tile * tile];
     __shared__ float B_buf[tile * tile];

     float ans = 0;
     for (int z_chip = 0; z_chip * tile < k; ++z_chip) {
         A_chip = &i4(A, ibx, z_chip, tile * k, tile);
         B_chip = &i4(B, z_chip, iby, tile * n, tile);
         i4(A_buf, itx, ity, tile, 1) = i4(A_chip, itx, ity, k, 1);
         i4(B_buf, itx, ity, tile, 1) = i4(B_chip, itx, ity, n, 1);
         __syncthreads();

         for (int z = 0; z < tile; ++z) {
             ans += i4(A_buf, itx, z, tile, 1) * i4(B_buf, z, ity, tile, 1);
         }
         __syncthreads();
     }

     C_chip = &i4(C, ibx, iby, tile * n, tile);
     i4(C_chip, itx, ity, n, 1) = ans;
}

// 单个线程计算更多
template <int tile = 64, int step = 2>
__global__
void matmul_v3(float* A, float* B, float* C, int m, int n, int k) {
    float* A_chip = NULL;
    float* B_chip = NULL;
    float* C_chip = NULL;
    __shared__ float A_buf[tile * tile];
    __shared__ float B_buf[tile * tile];
    __shared__ float C_buf[step * step];

    for (int z_chip = 0; z_chip * tile < k; ++z_chip) {
        A_chip = &i4(A, ibx, z_chip, tile * k, tile);
        B_chip = &i4(B, z_chip, iby, tile * n, tile);

        for (int step_x = 0; step_x < step; ++step_x) {
            for (int step_y = 0; step_y < step; ++step_y) {
                const int tile_x = itx * step + step_x, tile_y = ity * step + step_y;
                i4(A_buf, tile_x, tile_y, tile, 1) = i4(A_chip, tile_x, tile_y, k, 1);
                i4(B_buf, tile_x, tile_y, tile, 1) = i4(B_chip, tile_x, tile_y, n, 1);
            }
        }
        __syncthreads();

        for (int z = 0; z < tile; ++z) {
            for (int step_x = 0; step_x < step; ++step_x) {
                for (int step_y = 0; step_y < step; ++step_y) {
                    const int tile_x = itx * step + step_x, tile_y = ity * step + step_y;
                    i4(C_buf, step_x, step_y, step, 1) += i4(A_buf, tile_x, z, tile, 1) * i4(B_buf, z, tile_y, tile, 1);
                }
            }
        }
        __syncthreads();
    }

    C_chip = &i4(C, ibx, iby, tile * n, tile);
    for (int step_x = 0; step_x < step; ++step_x) {
        for (int step_y = 0; step_y < step; ++step_y) {
            const int tile_x = itx * step + step_x, tile_y = ity * step + step_y;
            i4(C_chip, tile_x, tile_y, n, 1) = i4(C_buf, step_x, step_y, step, 1);
        }
    }
}

// 单个线程计算更多，使用float4加载数据
template <int tile = 64, int step_m = 1, int step_n = 4>
__global__
void matmul_v4(float* A, float* B, float* C, int m, int n, int k) {
    static_assert(step_n == 4);
    float* A_chip = NULL;
    float* B_chip = NULL;
    float* C_chip = NULL;
    __shared__ float A_buf[tile * tile];
    __shared__ float B_buf[tile * tile];
    __shared__ float C_buf[step_m * step_n];

    for (int z_chip = 0; z_chip * tile < k; ++z_chip) {
        A_chip = &i4(A, ibx, z_chip, tile * k, tile);
        B_chip = &i4(B, z_chip, iby, tile * n, tile);

        for (int step_x = 0; step_x < step_m; ++step_x) {
            const int tile_x = itx * step_m + step_x, tile_y = ity * step_n;
            vload(i4(A_buf, tile_x, tile_y, tile, 1), &i4(A_chip, tile_x, tile_y, k, 1));
            vload(i4(B_buf, tile_x, tile_y, tile, 1), &i4(B_chip, tile_x, tile_y, n, 1));
        }
        __syncthreads();

        for (int z = 0; z < tile; ++z) {
            for (int step_x = 0; step_x < step_m; ++step_x) {
                for (int step_y = 0; step_y < step_n; ++step_y) {
                    const int tile_x = itx * step_m + step_x, tile_y = ity * step_n + step_y;
                    i4(C_buf, step_x, step_y, step_n, 1) += i4(A_buf, tile_x, z, tile, 1) * i4(B_buf, z, tile_y, tile, 1);
                }
            }
        }
        __syncthreads();
    }

    // TODO 后面应该重新定义指针指向的内存类型，直接取数，而不应该使用i4这种宏
    C_chip = &i4(C, ibx, iby, tile * n, tile);
    for (int step_x = 0; step_x < step_m; ++step_x) {
        const int tile_x = itx * step_m + step_x, tile_y = ity * step_n;
        // vload(i4(C_chip, tile_x, tile_y, n, 1), &i4(C_buf, tile_x, tile_y, step_n, 1));
        // NOTE ##############################################
    }
}

// 使用寄存器分块
template <int tile = 64, int step_m = 4, int step_n = 4>
__global__
void matmul_v5(float* A, float* B, float* C, int m, int n, int k) {
    static_assert(step_n == 4);
    float* A_chip = NULL;
    float* B_chip = NULL;
    float* C_chip = NULL;
    __shared__ float A_buf[tile * tile];
    __shared__ float B_buf[tile * tile];
    __shared__ float C_buf[step_m * step_n];
    float A_reg[step_m];
    float B_reg[step_n];
    float C_reg[step_m][step_n];

    for (int step_x = 0; step_x < step_m; ++step_x) {
        for (int step_y = 0; step_y < step_n; ++step_y) {
            C_reg[step_x][step_y] = 0;
        }
    }

    for (int z_chip = 0; z_chip * tile < k; ++z_chip) {
        A_chip = &i4(A, ibx, z_chip, tile * k, tile);
        B_chip = &i4(B, z_chip, iby, tile * n, tile);

        for (int step_x = 0; step_x < step_m; ++step_x) {
            const int tile_x = itx * step_m + step_x, tile_y = ity * step_n;
            vload(i4(A_buf, tile_x, tile_y, tile, 1), &i4(A_chip, tile_x, tile_y, k, 1));
            vload(i4(B_buf, tile_x, tile_y, tile, 1), &i4(B_chip, tile_x, tile_y, n, 1));
        }
        __syncthreads();

        for (int z = 0; z < tile; ++z) {
            for (int step_x = 0; step_x < step_m; ++step_x) A_reg[step_x] = i4(A_buf, step_x, z, tile, 1);
            for (int step_y = 0; step_y < step_n; ++step_y) B_reg[step_y] = i4(B_buf, z, step_y, tile, 1);

            for (int step_x = 0; step_x < step_m; ++step_x) {
                for (int step_y = 0; step_y < step_n; ++step_y) {
                    C_reg[step_x][step_y] += A_reg[step_x] * B_reg[step_y];
                }
            }
        }
        __syncthreads();
    }


    for (int step_x = 0; step_x < step_m; ++step_x) {
        for (int step_y = 0; step_y < step_n; ++step_y) {
            const int tile_x = itx * step_m + step_x, tile_y = ity * step_n + step_y;
            i4(C_chip, tile_x, tile_y, n, 1) = C_reg[step_x][step_y];
        }
    }

}




void check(Mat* A, Mat* B, Mat* C) {
    int m = A->m, n = B->n, k = A->n;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float a = 0;
            for (int ij = 0; ij < k; ++ij) {
                a += i2(A->h, A->n, i, ij) * i2(B->h, B->n, ij, j);
                // printf("------> %f\n", i2(A->h, A->n, i, ij) * i2(B->h, B->n, ij, j));
            }
            // if (a != i2(C->h, C->n, i, j)) {
            float diff = a - i2(C->h, C->n, i, j);
            // if (a == i2(C->h, C->n, i, j)) {
            if (abs(diff) > 0.001) {
                // A->print();
                // B->print();
                // C->print();
                printf("in [%d][%d], %f is fail, %f is ok, %f is diff\n", i, j, i2(C->h, C->n, i, j), a, diff);
                printf("A has \n");
                printf("[");
                for (int ij = 0; ij < k; ++ij) printf("%f, ", A->index(i, ij, true));
                printf("]\n");
                printf("B has \n");
                printf("[");
                for (int ij = 0; ij < k; ++ij) printf("%f, ", B->index(ij, j, true));
                printf("]\n");
                exit(0);
            }
        }
    }
}


const int m = 32 * 40;
const int n = 32 * 40;
const int k = 32 * 40;

struct Timer {
    cudaEvent_t begin, end;
    Timer() {
        cuda_check(cudaEventCreate(&begin));
        cuda_check(cudaEventCreate(&end));
        cudaEventRecord(begin);
    }
    ~Timer() {
        cudaEventRecord(end);
        cudaEventSynchronize(begin);
        cudaEventSynchronize(end);

        float time = 0;
        cudaEventElapsedTime(&time, begin, end);
        printf("time=%f ms, gfloats=%f\n", time, 2.0 * m * n * k * 1e-9 / time);

        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
};

int main() {
    srand(42);

    Mat A("A", m, k);
    Mat B("B", k, n);
    Mat C("C", m, n);
    A.init();
    B.init();
    A.move(true);
    B.move(true);

    {
        Timer t;
        cudaDeviceSynchronize();
        matmul_v2<<<dim3(m / 32, n / 32), dim3(32, 32)>>>((float*)A.d, (float*)B.d, (float*)C.d, m, n, k);
        cudaDeviceSynchronize();
    }


    C.move(false);
    check(&A, &B, &C);
    return 0;
}