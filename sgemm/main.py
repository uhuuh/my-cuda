# test_sgemm.py
import os
import torch
import matplotlib.pyplot as plt
import statistics

from torch.utils.cpp_extension import load
from contextlib import contextmanager
from types import SimpleNamespace
from tqdm import tqdm


# ===============================
# CUDA timer
# ===============================
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


# ===============================
# SGEMM operator loader
# ===============================
_OP_CACHE = {}

def load_sgemm_op(op_type: str, debug=False):
    if op_type in _OP_CACHE:
        return _OP_CACHE[op_type]

    cc_major, cc_minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc_major}.{cc_minor}"

    build_root = os.path.join(os.getcwd(), "build")
    build_subdir_name = f"sgemm_op={op_type}___debug={debug}___"
    build_dir = os.path.join(build_root, build_subdir_name)
    os.makedirs(build_dir, exist_ok=True)

    extra_cuda_cflags = ["-O3", "-lineinfo"] if not debug else ["-O0", "-lineinfo", "-G", "-g"]
    lib = load(
        name=f"{op_type}",
        sources=[f"{op_type}.cu"],
        build_directory=build_dir,
        verbose=True,
        extra_cuda_cflags=extra_cuda_cflags,
    )

    op = getattr(lib, f"{op_type}")
    _OP_CACHE[op_type] = op
    return op


# ===============================
# PyTorch baseline GEMM
# ===============================
def torch_gemm(A, B):
    """Standard PyTorch matrix multiplication"""
    return torch.matmul(A, B)


# ===============================
# Refactored Benchmark Functions
# ===============================

def benchmark_single_run(op_fn, A, B, warmup=5, samples=10, reference_C=None):
    """
    在给定输入上执行算子并记录时间
    """
    # 预热
    for _ in range(warmup):
        op_fn(A, B)

    # 计时采样
    times = []
    for _ in range(samples):
        with cuda_timer() as t:
            C = op_fn(A, B)
        times.append(t.ms)

    # 验证正确性 (若提供参考结果)
    if reference_C is not None:
        max_diff = (C - reference_C).abs().max().item()
        # 对于 FP32，1e-2 是一个比较通用的阈值
        assert max_diff < 1e-2, f"精度校验失败! Max diff: {max_diff}"

    return {"times": times, "result": C if reference_C is None else None}


def run_benchmark_plot(baseline_fn, ops_dict, warmup=5, samples=10):
    """
    按尺寸迭代：每个尺寸下共享相同的 A, B 输入
    """
    GROUPS = {
        "Square GEMM": [
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ],
        "M,N large / K small": [
            (8192, 1024, 8192),
            (16384, 1024, 8192),
        ],
        "M small / K,N large": [
            (1024, 8192, 8192),
            (1024, 16384, 8192),
        ],
    }

    op_names = list(ops_dict.keys())
    # 结构: {group_name: {impl_name: {(M,K,N): {times:[], gflops:float}}}}
    all_results = {group: {name: {} for name in (["torch"] + op_names)} for group in GROUPS}

    print("\n🚀 开始性能基准测试 (共享输入模式)...")

    for group_name, shapes in GROUPS.items():
        print(f"\n测试组: {group_name}")

        for shape in tqdm(shapes, desc="Processing Shapes"):
            M, K, N = shape
            # 1. 准备该尺寸下共享的输入
            A = torch.randn(M, K, device="cuda")
            B = torch.randn(K, N, device="cuda")
            gflops_val = 2.0 * M * K * N / 1e9

            # 2. 运行 PyTorch 基准获取参考结果和时间
            torch_res = benchmark_single_run(baseline_fn, A, B, warmup, samples)
            all_results[group_name]["torch"][shape] = {
                "times": torch_res["times"],
                "gflops": gflops_val
            }
            ref_C = torch_res["result"]

            # 3. 运行各个自定义算子并对比
            for op_name, op_fn in ops_dict.items():
                try:
                    res = benchmark_single_run(op_fn, A, B, warmup, samples, reference_C=ref_C)
                    all_results[group_name][op_name][shape] = {
                        "times": res["times"],
                        "gflops": gflops_val
                    }
                except Exception as e:
                    print(f"\n❌ Error in {op_name} @ {shape}: {e}")

            # 4. 清理显存避免 OOM
            del A, B, ref_C
            torch.cuda.empty_cache()

    print("\n📊 绘图中...")
    plot_all_groups(all_results, op_names)


# ===============================
# Plot: ONE FIGURE, MULTI SUBPLOTS
# ===============================
def plot_all_groups(all_results, op_names):
    def compute_stats(results):
        stats = {"x": [], "mean": [], "min": [], "max": []}
        for shape in sorted(results.keys()):
            data = results[shape]
            gflops = data["gflops"]
            perfs = [gflops / (t / 1e3) for t in data["times"]]
            stats["x"].append(gflops)
            stats["mean"].append(statistics.mean(perfs))
            stats["min"].append(min(perfs))
            stats["max"].append(max(perfs))
        return stats

    num_groups = len(all_results)
    cols = 2
    rows = (num_groups + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False, sharey=True)

    styles = {"torch": {"fmt": "s--", "color": "black", "label": "PyTorch"}}
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, op_name in enumerate(op_names):
        styles[op_name] = {"fmt": "o-", "color": colors[i % len(colors)], "label": op_name.replace("_", " ").title()}

    for ax, (group_name, group_results) in zip(axes.flat, all_results.items()):
        for impl_name in ["torch"] + op_names:
            if not group_results[impl_name]: continue
            stats = compute_stats(group_results[impl_name])
            yerr = [[stats["mean"][i] - stats["min"][i] for i in range(len(stats["mean"]))],
                    [stats["max"][i] - stats["mean"][i] for i in range(len(stats["mean"]))]]

            ax.errorbar(stats["x"], stats["mean"], yerr=yerr, fmt=styles[impl_name]["fmt"],
                        color=styles[impl_name]["color"], capsize=4, label=styles[impl_name]["label"],
                        linewidth=2, markersize=7 if impl_name=="torch" else 6)

        ax.set_title(group_name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Problem size (GFLOPs)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for ax in axes.flat[num_groups:]: ax.set_visible(False)
    axes[0, 0].set_ylabel("Throughput (GFLOPs/s)", fontsize=10)
    fig.suptitle("SGEMM Performance Comparison (Shared Inputs)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig("sgemm_benchmark.png", dpi=150, bbox_inches='tight')
    plt.show()


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        exit()

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    debug = False
    op_types = ["sgemm_native", "sgemm_smem_cache", "sgemm_rmem_cache", "sgemm_double_buffer", "sgemm_memory_coalesce"]

    # 加载算子
    ops_dict = {}
    print("Loading operators...")
    for op_type in op_types:
        print(f"  - Loading {op_type}...")
        ops_dict[op_type] = load_sgemm_op(op_type, debug)

    # 运行基准测试
    run_benchmark_plot(
        baseline_fn=torch_gemm,
        ops_dict=ops_dict,
        warmup=5,
        samples=10
    )