# test_sgemm.py
import os
import time
import torch
from torch.utils.cpp_extension import load
from contextlib import contextmanager
from types import SimpleNamespace


@contextmanager
def cuda_timer():
    result = SimpleNamespace(ms=None)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()


    start.record()
    yield result
    end.record()

    torch.cuda.synchronize()
    result.ms = start.elapsed_time(end)


# 设置编译目录
build_dir = os.path.join(os.getcwd(), 'build')
os.makedirs(build_dir, exist_ok=True)

cc_major, cc_minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc_major}.{cc_minor}"

# 动态编译CUDA算子
sgemm_module = load(
    name="sgemm_smem_cache",
    sources=["sgemm_smem_cache.cu"],
    build_directory=build_dir,
    verbose=True,
    is_python_module=False,
    extra_cuda_cflags=["-O3", '-lineinfo'],
)

def test_performance():
    # 配置测试参数
    M, K, N = 2048, 2048, 2048
    device = torch.device('cuda')

    # 生成随机测试数据
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    # 预热
    for _ in range(5):
        C = torch.ops.my.sgemm_smem_cache(A, B)

    with cuda_timer() as timer:
        C = torch.ops.my.sgemm_smem_cache(A, B)

    C_ref = torch.matmul(A, B)
    max_diff = (C - C_ref).abs().max().item()
    print(f"Max difference vs PyTorch: {max_diff:.6e}, Time: {timer.ms:.6f} ms")
    assert max_diff < 1e-3, f"Result mismatch! Max diff: {max_diff}"

if __name__ == "__main__":
    # 检查CUDA设备
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")

    # 运行性能测试
    test_performance()