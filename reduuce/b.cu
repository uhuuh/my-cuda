// #define enable_debug
#include "../sgemm/util.h"

const int m = 256 * 20000;
const int block = 256;
const int tile = block;

__global__
void reduce(float* v_in, float* v_out) {
    __shared__ float sdata[tile];
    v_in += ibx * tile;
    sdata[itx] = v_in[itx];
    __syncthreads();
    debug(i_ == 0, "x0=%f, x1=%f, x2=%f, x3=%f", sdata[0], sdata[1], sdata[2], sdata[3]);

    for (int x = 1; x < tile; x <<= 1) {
        if (itx % (x << 1) == 0) { // 一个位置上聚合两个步长的结果，因此下一个位置在两个步长后
            sdata[itx] += sdata[itx + x];
            __syncthreads(); // 为什么这里需要同步? 当步长超过一个warp的时候，需要插入sync保证不同warp间执行正确
            // TODO cuda中kernel中使用print， 会有同步效果吗
            debug(i_ == 0, "[0] step=%d, val=%f", x, sdata[itx]);
            debug(i_ == 2, "[2] step=%d, val=%f", x, sdata[itx]);
        }
    }

    if (itx == 0) {
        v_out[ibx] = sdata[0];
    }
}

// 消除warp分支，将相同分支的线程聚合在一个warp中
__global__
void reduce_v2(float* v_in, float* v_out) {
    __shared__ float sdata[tile];
    v_in += ibx * tile;
    sdata[itx] = v_in[itx];
    __syncthreads();

    for (int x = 1; x < tile; x <<= 1) {
        int step = x << 1;
        if (itx < tile / step) {
            sdata[itx * step] += sdata[itx * step + x];
            __syncthreads();
        }
    }

    if (itx == 0) {
        v_out[ibx] = sdata[0];
    }
}

// 解决bank冲突
__global__
void reduce_v3(float* v_in, float* v_out) {
    __shared__ float sdata[tile];
    v_in += ibx * tile;
    sdata[itx] = v_in[itx];
    __syncthreads();

    for (int step = tile / 2; step > 0; step >>= 1) {
        if (itx < step) {
            sdata[itx] += sdata[itx + step];
            __syncthreads();
        }
    }

    if (itx == 0) {
        v_out[ibx] = sdata[0];
    }
}

// 所有线程增加计算
__global__
void reduce_v4(float* v_in, float* v_out) {
    __shared__ float sdata[tile];
    v_in += ibx * tile;
    // NOTE 这里可以多计算一些
    sdata[itx] = v_in[itx] + v_in[itx + tile]; // 相比于v3修改这一行，注意线程数量需要特殊设置
    __syncthreads();

    for (int step = tile / 2; step > 0; step >>= 1) {
        if (itx < step) {
            sdata[itx] += sdata[itx + step];
            __syncthreads();
        }
    }

    if (itx == 0) {
        v_out[ibx] = sdata[0];
    }
}

// 减少不必要的同步
__global__
void reduce_v5(float* v_in, float* v_out) {
    __shared__ float sdata[tile];
    v_in += ibx * tile;
    sdata[itx] = v_in[itx] + v_in[itx + tile]; // 相比于v3修改这一行，注意线程数量需要特殊设置
    __syncthreads();

    for (int step = tile / 2; step > 0; step >>= 1) {
        if (itx < step) {
            sdata[itx] += sdata[itx + step];
            if (step > 32) __syncthreads();
        }
    }

    if (itx == 0) {
        v_out[ibx] = sdata[0];
    }
}

// 循环展开
__global__
void reduce_v6(float* v_in, float* v_out) {
    __shared__ float sdata[tile];
    v_in += ibx * tile;
    sdata[itx] = v_in[itx] + v_in[itx + tile]; // 相比于v3修改这一行，注意线程数量需要特殊设置
    __syncthreads();

    // TODO 不确定下面是否生效
#pragma unroll
    for (int step = tile / 2; step > 0; step >>= 1) {
        if (itx < step) {
            sdata[itx] += sdata[itx + step];
            if (step > 32) __syncthreads();
        }
    }

    if (itx == 0) {
        v_out[ibx] = sdata[0];
    }
}

// 通过shuffle来reduce, block数量小于1024
template <int block, int warp = 32>
__global__
void reduce_v7(float* v_in, float* v_out) {
    static_assert(warp == 32, "warp must is 32");
    static_assert(block <= warp * warp, "block must < warp * warp");

    __shared__ float sdata[warp];
    int warp_id = itx / warp;
    int lane_id = itx % warp;

    v_in += ibx * tile;
    float sum = v_in[itx] + v_in[itx + tile]; // 相比于v3修改这一行，注意线程数量需要特殊设置
    __syncthreads();

    __shfl_down_sync(0xffffffff, sum, 16);
    __shfl_down_sync(0xffffffff, sum, 8);
    __shfl_down_sync(0xffffffff, sum, 4);
    __shfl_down_sync(0xffffffff, sum, 2);
    __shfl_down_sync(0xffffffff, sum, 1);
    if (block <= warp) {
        if (itx == 0) v_out[itx] = sum;
        return;
    }

    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    sum = itx < warp ? sdata[itx] : 0;

    __shfl_down_sync(0xffffffff, sum, 16);
    __shfl_down_sync(0xffffffff, sum, 8);
    __shfl_down_sync(0xffffffff, sum, 4);
    __shfl_down_sync(0xffffffff, sum, 2);
    __shfl_down_sync(0xffffffff, sum, 1);

    if (itx == 0) v_out[ibx] = sum;
}

void reduce_cpu(float* arr, float* out, int m, int block) {
    assert(m % block == 0);
    for (int i = 0; i < m; i += block) {
        float sum = 0;
        for (int j = 0; j < block; ++j) {
            sum += arr[i + j];
        }
        out[i / block] = sum;
    }
}

int main() {
    srand(42);

    Mat v_in(true, 1, 1, m);
    Mat v_out(true, 1, 1, m / block);
    v_in.init();
    debug(true, "input: %s\n", v_in.str().c_str());

    v_in.move(true);
    cudaDeviceSynchronize();

    {
        Timer timer(m);
        reduce_v3<<<dim3(m / block), dim3(block)>>>(v_in.device, v_out.device);
        cudaDeviceSynchronize();
    }

    v_out.move(false);
    cudaDeviceSynchronize();

    Mat v_in2(false, 1, 1, m);
    Mat v_out2(false, 1, 1, m / block);
    v_in2.init(&v_in);
    reduce_cpu(v_in2.host, v_out2.host, m, block);

    if (!v_out.equ(&v_out2)) {
        printf("error\n");
        printf("now    : %s\n", v_out.str().c_str());
        printf("correct: %s\n", v_out2.str().c_str());
    } else {
        printf("ok\n");
    }

    return 0;
}