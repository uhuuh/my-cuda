# test_sgemm.py
import os
import torch
import matplotlib.pyplot as plt
import statistics
import numpy as np

from torch.utils.cpp_extension import load
from contextlib import contextmanager
from types import SimpleNamespace
from tqdm import tqdm

# ===============================
# 1. CUDA 计时器
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
# 2. CUDA 算子加载器 (JIT 编译)
# ===============================
_OP_CACHE = {}

def load_op(op_type: str, debug=False):
    """加载并编译指定的 CUDA 算子"""
    if op_type in _OP_CACHE:
        return _OP_CACHE[op_type]

    cc_major, cc_minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc_major}.{cc_minor}"

    build_root = os.path.join(os.getcwd(), "build")
    build_subdir_name = f"op={op_type}___debug={debug}___"
    build_dir = os.path.join(build_root, build_subdir_name)
    os.makedirs(build_dir, exist_ok=True)

    extra_cuda_cflags = ["-O3", "-lineinfo"] if not debug else ["-O0", "-lineinfo", "-G", "-g"]

    # 假设源文件命名为 sgemm_{op_type}.cu
    source_file = f"{op_type}.cu"
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"找不到源文件: {source_file}")

    lib = load(
        name=f"{op_type}",
        sources=[source_file],
        build_directory=build_dir,
        verbose=True,
        extra_cuda_cflags=extra_cuda_cflags,
    )

    op = getattr(lib, f"{op_type}")
    _OP_CACHE[op_type] = op
    return op

# ===============================
# 3. 核心基准测试逻辑 (重构版)
# ===============================
def benchmark_single_run(op_fn, A, B, warmup=5, samples=10, reference_C=None):
    """在给定输入 A, B 上测试算子性能并验证结果"""
    # 预热
    for _ in range(warmup):
        op_fn(A, B)

    # 计时
    times = []
    for _ in range(samples):
        with cuda_timer() as t:
            C = op_fn(A, B)
        times.append(t.ms)

    # 正确性验证
    if reference_C is not None:
        max_diff = (C - reference_C).abs().max().item()
        # 对于 FP32，允许一定的浮点误差
        assert max_diff < 1e-2, f"精度验证失败! Max diff: {max_diff:.6e}"

    return {"times": times, "result": C if reference_C is None else None}

def run_benchmark_plot(baseline_fn, ops_dict, warmup=5, samples=10):
    """尺寸优先测试：每个 shape 下共享输入，执行所有算子"""
    GROUPS = {
        "Square GEMM": [
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ],
        "M,N Large / K Small": [
            (8192, 1024, 8192),
            (16384, 1024, 8192),
        ],
        "M Small / K,N Large": [
            (1024, 8192, 8192),
            (1024, 16384, 8192),
        ],
    }

    op_names = list(ops_dict.keys())
    # 存储结构: {group_name: {impl_name: {(M,K,N): {times:[], gflops:float}}}}
    all_results = {group: {name: {} for name in (["torch"] + op_names)} for group in GROUPS}

    print("\n🚀 开始 SGEMM 统一基准测试 (共享输入模式)...")

    for group_name, shapes in GROUPS.items():
        print(f"\n测试组: {group_name}")
        for shape in tqdm(shapes, desc="正在处理不同尺寸"):
            M, K, N = shape

            # --- 1. 准备该尺寸下共享的输入 ---
            A = torch.randn(M, K, device="cuda")
            B = torch.randn(K, N, device="cuda")
            gflops_val = 2.0 * M * K * N / 1e9

            # --- 2. 先运行 PyTorch (cuBLAS) 获取参考时间与结果 ---
            torch_res = benchmark_single_run(baseline_fn, A, B, warmup, samples)
            all_results[group_name]["torch"][shape] = {
                "times": torch_res["times"],
                "gflops": gflops_val
            }
            ref_C = torch_res["result"]

            # --- 3. 运行所有自定义算子并进行验证 ---
            for op_name, op_fn in ops_dict.items():
                try:
                    res = benchmark_single_run(op_fn, A, B, warmup, samples, reference_C=ref_C)
                    all_results[group_name][op_name][shape] = {
                        "times": res["times"],
                        "gflops": gflops_val
                    }
                except Exception as e:
                    print(f"\n❌ 算子 {op_name} 在尺寸 {shape} 出错: {e}")

            # --- 4. 显存清理防止 OOM ---
            del A, B, ref_C
            torch.cuda.empty_cache()

    # 执行绘图
    plot_all_groups(all_results, op_names)

# ===============================
# 4. 可视化函数 (增强版)
# ===============================
def plot_all_groups(all_results, op_names):
    """绘制高对比度图表，强化列坐标网格"""
    def compute_stats(results):
        stats = {"x": [], "mean": [], "min": [], "max": []}
        for shape in sorted(results.keys()):
            data = results[shape]
            gflops = data["gflops"]
            # 性能计算: GFLOPs / (ms / 1000)
            perfs = [gflops / (t / 1e3) for t in data["times"]]
            stats["x"].append(gflops)
            stats["mean"].append(statistics.mean(perfs))
            stats["min"].append(min(perfs))
            stats["max"].append(max(perfs))
        return stats

    num_groups = len(all_results)
    cols = 2
    rows = (num_groups + cols - 1) // cols

    # 配色方案：鲜艳的霓虹色
    vibrant_colors = ["#FF0055", "#00E676", "#2979FF", "#FF9100", "#D500F9", "#00E5FF"]
    styles = {"torch": {"fmt": "s--", "color": "#000000", "label": "PyTorch (cuBLAS)"}}
    for i, name in enumerate(op_names):
        styles[name] = {
            "fmt": "o-",
            "color": vibrant_colors[i % len(vibrant_colors)],
            "label": name.replace("_", " ").title()
        }

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6), squeeze=False)
    fig.patch.set_facecolor('#F8F9FA')

    for ax, (group_name, group_results) in zip(axes.flat, all_results.items()):
        ax.set_facecolor('#FFFFFF')

        # --- 强化网格与列坐标 ---
        ax.minorticks_on()
        # 主网格线（主要坐标）
        ax.grid(True, which='major', linestyle='-', linewidth=0.8, color='#D1D1D1')
        # 次网格线（即“列坐标”感）
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='#EAEAEA')

        for impl_name in ["torch"] + op_names:
            if not group_results[impl_name]: continue
            stats = compute_stats(group_results[impl_name])

            yerr = [[stats["mean"][i] - stats["min"][i] for i in range(len(stats["mean"]))],
                    [stats["max"][i] - stats["mean"][i] for i in range(len(stats["mean"]))]]

            ax.errorbar(
                stats["x"], stats["mean"], yerr=yerr,
                fmt=styles[impl_name]["fmt"], color=styles[impl_name]["color"],
                capsize=5, capthick=1.5, label=styles[impl_name]["label"],
                linewidth=2.5, markersize=8, alpha=0.9
            )

        ax.set_title(group_name, fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel("Workload (GFLOPs)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Throughput (GFLOPs/s)", fontsize=11, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)

    # 隐藏空子图
    for ax in axes.flat[num_groups:]:
        ax.set_visible(False)

    fig.suptitle("SGEMM Performance Comparison (Shared Inputs per Shape)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = "sgemm_benchmark_vibrant.png"
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    print(f"\n✅ 测试完成！图表已保存至: {out_file}")
    plt.show()

# ===============================
# 5. 主程序入口
# ===============================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误: 未检测到 CUDA 设备")
        exit()

    print(f"当前 GPU 设备: {torch.cuda.get_device_name(0)}")

    # 调试模式标志
    is_debug = False

    # 定义要测试的算子文件名（不含 .cu 后缀）
    op_types = ["sgemm_native", "sgemm_smem_cache", "sgemm_rmem_cache", "sgemm_double_buffer", "sgemm_memory_coalesce"]

    ops_dict = {}
    print("\n正在编译并加载 CUDA 算子...")
    for op_type in op_types:
        try:
            ops_dict[op_type] = load_op(op_type, debug=is_debug)
            print(f"  - {op_type} 加载成功")
        except Exception as e:
            print(f"  - {op_type} 加载失败: {e}")

    if not ops_dict:
        print("未加载任何自定义算子，仅运行 PyTorch 基准。")

    # 运行完整的性能分析
    run_benchmark_plot(
        baseline_fn=torch.matmul, # 直接使用 torch.matmul 作为基准
        ops_dict=ops_dict,
        warmup=5,
        samples=10
    )