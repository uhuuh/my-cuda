from torch.utils.cpp_extension import load
import torch
from contextlib import contextmanager
import triton
import triton.language as tl
def fun_time(fun):
    def new_fun(*arg, **kwarg):
        iter = 10
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []

        for _ in range(iter):
            start.record()
            fun(*arg, **kwarg)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            times.append(ms)
            print("--------->", ms)
        print(fun.__name__, times)
    return new_fun

module = load(
    name="my_extension",
    sources=["a.cu"],  # 你的源码
    verbose=True,
    is_python_module=False,
    extra_cuda_cflags=['-arch=sm_120 -lineinfo'],
)

def test_add():
    a = torch.randn(5, device="cuda")
    b = torch.randn(5, device="cuda")

    #out = module.my_add(a, b)
    out = torch.ops.my.add(a, b)
    print(out)

def test_nan_to_num():
    a = torch.empty((32000, 8000), device="cuda")
    a[1] = float("nan")

    @fun_time
    def test_nan_to_num_v1():
        b = torch.ops.my.nan_to_num(a)

    @fun_time
    def test_nan_to_num_v2():
        b = torch.ops.my.nan_to_num_v2(a)

    test_nan_to_num_v1()
    # test_nan_to_num_v2()

def test_mem():
    a = torch.randn((32000, 8000), device="cuda")

    @fun_time
    def test_mem():
        torch.ops.my.test_mem(a)

    test_mem()

def test_triton():
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
            triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        ],
        key=['size'],  # 根据输入大小选择配置
    )
    @triton.jit
    def nan_to_num_kernel(
        x_ptr,
        size: tl.constexpr,
        nan_val: tl.constexpr = 0.0,
        posinf_val: tl.constexpr = 1e20,
        neginf_val: tl.constexpr = -1e20,
        BLOCK_SIZE: tl.constexpr = 1024
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < size

        x = tl.load(x_ptr + offs, mask=mask, other=0.0)

        # 替换 NaN：NaN != NaN 为 True
        is_nan = x != x
        x = tl.where(is_nan, nan_val, x)

        # 替换 +inf：x > max_float -> +inf
        is_posinf = x > 3.4e38
        x = tl.where(is_posinf, posinf_val, x)

        # 替换 -inf：x < min_float -> -inf
        is_neginf = x < -3.4e38
        x = tl.where(is_neginf, neginf_val, x)

        tl.store(x_ptr + offs, x, mask=mask)

    # Python wrapper
    @fun_time
    def nan_to_num(x: torch.Tensor, nan_val=0.0, posinf_val=1e20, neginf_val=-1e20):
        assert x.is_cuda
        x_flat = x.flatten()
        size = x_flat.numel()
        grid = (triton.cdiv(size, 1024),)
        y = nan_to_num_kernel[grid](x_flat, size, nan_val, posinf_val, neginf_val)
        # print(y.asm['ptx'])
        return x

    a = torch.randn((32000, 8000), device="cuda")
    a[1] = float("nan")
    nan_to_num(a)

def test_torch():
    a = torch.randn((32000, 8000), device="cuda")
    @fun_time
    def test_torch():
        b = torch.nan_to_num(a)
    test_torch()

test_nan_to_num()
# test_triton()
# test_torch()
# test_triton()



