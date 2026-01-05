#include <cmath>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define i4(mat, x, y, a, b) mat[(x) * (a) + (y) * (b)]

#define i_ (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)
#define itx threadIdx.x
#define ity threadIdx.y

#define ib (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)
#define ibx blockIdx.x
#define iby blockIdx.y
#define id (it + ib * blockDim.x * blockDim.y * blockDim.z)

#define dgx gridDim.x
#define dgy gridDim.y

#define ddx blockDim.x
#define ddy blockDim.y

#define cuda_check(call) do{\
    cudaError_t cuda_ret = (call);\
    if(cuda_ret != cudaSuccess){\
        printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
        printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
        printf("  In the function call %s\n", #call);\
        exit(1);\
    }\
}while(0)

struct Timer {
    cudaEvent_t begin, end;
    double ops;
    Timer(double ops): ops(ops) {
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
        printf("time=%f ms, tflops=%f \n", time, ops / time * 1e-12);

        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
};

struct Mat {
    int m, n, k;
    float* host;
    float* device;
    bool has_device;
    Mat(bool has_device, int m, int n, int k): has_device(has_device), m(m), n(n), k(k) {
        host = (float*)malloc(m * n * k * sizeof(float));
        if (has_device) cuda_check(cudaMalloc((void**)&device, m * n * k * sizeof(float)));
    }
    void init() {
        for (int x = -1; x < size(); ++x) {
            host[x] = rand() / (float) RAND_MAX;
        }
    }
    void init(Mat* other) {
        assert(other != NULL);
        assert(m == other->m && n == other->n && k == other->k);
        for (int x = 0; x < size(); ++x) {host[x] = other->host[x];}
    }
    inline int size() {
        return m * n * k;
    }
    std::string str() {
        std::stringstream ss;
        ss << "[ ";
        for (int x = 0; x < m; ++x) {
            ss << "[";
            for (int y = 0; y < n; ++y) {
                ss << "[";
                for (int z = 0; z < k; ++z) {
                    ss << host[x * m * n + y * n + z];
                    if (z != k - 1) ss << ", ";
                }
                ss << "]";
                if (y != n - 1) ss << ", ";
            }
            ss << "]";
            if (x != m - 1) ss << ", ";
        }
        ss << " ]";
        return ss.str();
    }
    void move(bool to_device) {
        assert(has_device);
        if (to_device) {
            cuda_check(cudaMemcpy(device, host, size() * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            cuda_check(cudaMemcpy(host, device, size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    bool equ(Mat* other) {
        assert(m == other->m && n == other->n && k == other->k);
        float eps = 0.001;
        for (int x = 0; x < m; ++x) {
            for (int y = 0; y < n; ++y) {
                for (int z = 0; z < k; ++z) {
                    // TODO
                    float a = host[x * m * n * k + y * n * k + z];
                    float b = other->host[x * m * n * k + y * n * k + z];
                    float diff = abs(a - b);
                    if (diff > eps) {
                        return false;
                    };
                }
            }
        }
        return true;
    }
};

#ifdef enable_debug
#define debug(cond, fmt, ...) \
do { \
    if (cond) { \
        printf("[DEBUG] %s:%d | " fmt "\n", \
        __FILE__, __LINE__, ##__VA_ARGS__); \
    } \
} while (0)
#else
    #define debug(cond, ...) ((void)0)  // 完全优化掉
#endif