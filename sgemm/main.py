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
    build_subdir_name = (
        f"sgemm_op={op_type}___"
        f"debug={debug}___"
    )
    build_dir = os.path.join(build_root, build_subdir_name)
    os.makedirs(build_dir, exist_ok=True)

    extra_cuda_cflags = ["-O3", "-lineinfo"] if not debug else ["-O0", "-lineinfo", "-G", "-g"]
    load(
        name=f"sgemm_{op_type}",
        sources=[f"sgemm_{op_type}.cu"],
        build_directory=build_dir,
        verbose=True,
        is_python_module=False,
        extra_cuda_cflags=extra_cuda_cflags,
    )

    op = getattr(torch.ops.my, f"sgemm_{op_type}")
    _OP_CACHE[op_type] = op
    return op


# ===============================
# Benchmark single shape
# ===============================
def benchmark_shape(op, M, K, N, warmup=5, samples=10):
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")

    torch_fn = lambda: torch.matmul(A, B)
    custom_fn = lambda: op(A, B)

    for _ in range(warmup):
        torch_fn()
        custom_fn()

    torch.cuda.synchronize()

    torch_times, custom_times = [], []

    for _ in range(samples):
        with cuda_timer() as t:
            C_torch = torch_fn()
        torch_times.append(t.ms)

        with cuda_timer() as t:
            C_custom = custom_fn()
        custom_times.append(t.ms)

    max_diff = (C_custom - C_torch).abs().max().item()
    # TODO 大矩阵相乘，可能会有精度问题，累加器那里
    assert max_diff < 1e-2, f"Mismatch {max_diff} @ {M,K,N}"

    gflops = 2.0 * M * K * N / 1e9

    def stats(times):
        perfs = [gflops / (t / 1e3) for t in times]
        return {
            "mean": statistics.mean(perfs),
            "min": min(perfs),
            "max": max(perfs),
        }

    return {
        "gflops": gflops,
        "torch": stats(torch_times),
        "custom": stats(custom_times),
    }

# ===============================
# Plot: ONE FIGURE, MULTI SUBPLOTS
# ===============================
def plot_all_groups(all_results):
    num_groups = len(all_results)
    cols = 2
    rows = (num_groups + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 6, rows * 5),
        squeeze=False,
        sharey=True,
    )

    for ax, (group_name, results) in zip(axes.flat, all_results.items()):
        for impl, style in [("torch", "o--"), ("custom", "o-")]:
            x = results[impl]["x"]
            y = results[impl]["mean"]
            yerr = [
                [y[i] - results[impl]["min"][i] for i in range(len(y))],
                [results[impl]["max"][i] - y[i] for i in range(len(y))],
            ]
            ax.errorbar(x, y, yerr=yerr, fmt=style, capsize=4, label=impl)

        ax.set_title(group_name)
        ax.set_xlabel("Problem size (GFLOPs)")
        ax.grid(True)
        ax.legend()

    axes[0, 0].set_ylabel("Throughput (GFLOPs/s)")
    fig.tight_layout()
    plt.show()


# ===============================
# Shapes (ALL POWERS OF TWO)
# ===============================

def run_benchmark_plot(op):
    GROUPS = {
        "Square GEMM": [
            # (4, 4, 4),
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

    all_results = {}
    print("\nRunning benchmarks...")
    for group_name, shapes in tqdm(GROUPS.items(), desc="All groups"):
        results = {
            "torch": {"x": [], "mean": [], "min": [], "max": []},
            "custom": {"x": [], "mean": [], "min": [], "max": []},
        }

        for M, K, N in tqdm(shapes, desc=group_name, leave=False):
            r = benchmark_shape(op, M, K, N)
            for impl in ("torch", "custom"):
                results[impl]["x"].append(r["gflops"])
                results[impl]["mean"].append(r[impl]["mean"])
                results[impl]["min"].append(r[impl]["min"])
                results[impl]["max"].append(r[impl]["max"])

        all_results[group_name] = results
    print("\nPlotting...")
    plot_all_groups(all_results)


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    debug = False
    op_type = "memory_coalesce"
    # op_type = "double_buffer"
    # op_type = "rmem_cache"

    if debug:
        op = load_sgemm_op(op_type, debug)
        benchmark_shape(op, 512,512, 512)
        print(">>>>>> ok!!!")
    else:
        op = load_sgemm_op(op_type, debug)
        run_benchmark_plot(op)

