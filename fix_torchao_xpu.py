"""
fix_torchao_xpu.py — v4.0
──────────────────────────
torchao 0.17.0+xpu 的 experimental/ops/mps/ 模块在导入时抛出
RuntimeError（非 ImportError 子类），导致 pkgutil.walk_packages()
等模块发现工具在扫描 torchao 子包时崩溃。

本脚本移除该目录。仅对 +xpu 版本生效。
可安全重复运行。
"""
import os
import sys
import shutil


def main() -> int:
    # 1. 检测 torchao
    try:
        import torchao
    except ImportError:
        print("[ERROR] torchao 未安装。")
        print("  Intel XPU:  pip install torchao --index-url https://download.pytorch.org/whl/xpu")
        print("  NVIDIA/CUDA:  pip install torchao>=0.17.0")
        return 1

    ver = getattr(torchao, "__version__", "")
    if "+xpu" not in ver:
        print(f"[INFO] 标准 torchao {ver} 不需要此修复，跳过。")
        return 0

    pkg = os.path.dirname(os.path.abspath(torchao.__file__))
    mps_dir = os.path.join(pkg, "experimental", "ops", "mps")

    print(f"[INFO] torchao {ver} (XPU fork)")
    print(f"[INFO] 安装路径: {pkg}")

    if os.path.isdir(mps_dir):
        shutil.rmtree(mps_dir)
        print(f"\n[OK] 已移除: {mps_dir}")
        print("  重启 ComfyUI 后 diffusers / transformers 模块扫描恢复正常。")
    else:
        print("\n[OK] MPS 目录不存在，无需修复。")

    print("  本脚本可安全重复运行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
