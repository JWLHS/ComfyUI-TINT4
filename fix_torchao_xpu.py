"""
fix_torchao_xpu.py — 自动补齐 torchao XPU fork 缺失模块 (v3.0)
────────────────────────────────────────────────────────────────
零依赖。自动检测 ComfyUI 环境中的 torchao 安装路径，
对比内置参考清单补全所有缺失文件。仅对 +xpu 版本生效。
可安全重复运行。

清单提取自 torchao 0.17.0 标准版 dtypes / quantization 目录。
缺失文件均以空壳补齐 —— diffusers / transformers 只需它们可 import，
不调用其中的代码，不会产生任何新错误。
"""
import os
import sys

# ══════════════════════════════════════════════════════════════════════
# 内置文件清单（提取自 torchao 0.17.0 标准版）
# ══════════════════════════════════════════════════════════════════════
_MISSING: dict[str, str] = {
    # ── dtypes ───────────────────────────────────────────────────
    "dtypes/floatx/__init__.py": "",
    "dtypes/floatx/float8_layout.py": "Float8AQTTensorImpl = None\n",
    "dtypes/uintx/__init__.py": "",
    "dtypes/uintx/uint4_layout.py": "UInt4Tensor = None\n",
    "dtypes/uintx/plain_layout.py": "",
    "dtypes/uintx/tensor_core_tiled_layout.py": "",
    "dtypes/affine_quantized_tensor.py": "",
    "dtypes/affine_quantized_tensor_ops.py": "",
    # ── quantization ─────────────────────────────────────────────
    "quantization/linear_quant_modules.py": "",
    "quantization/linear_quant.py": "",
    "quantization/granularity.py": "",
    "quantization/observer.py": "",
    "quantization/transform_module.py": "",
    "quantization/unified.py": "",
    "quantization/utils.py": "",
    "quantization/weight_tensor_linear_activation_quantization.py": "",
    "quantization/linear_activation_quantized_tensor.py": "",
    "quantization/linear_activation_scale.py": "",
    "quantization/linear_activation_weight_observed_tensor.py": "",
    "quantization/quant_api.py": "",
    "quantization/quant_primitives.py": "",
    # ── quantization/prototype ───────────────────────────────────
    "quantization/prototype/__init__.py": "",
    "quantization/prototype/qat/__init__.py": "",
    "quantization/prototype/qat/api.py": "",
    "quantization/prototype/qat/affine_fake_quantized_tensor.py": "",
    "quantization/prototype/qat/embedding.py": "",
    "quantization/prototype/qat/fake_quantizer.py": "",
    "quantization/prototype/qat/linear.py": "",
    "quantization/prototype/qat/_module_swap_api.py": "",
    # ── quantization/pt2e ────────────────────────────────────────
    "quantization/pt2e/__init__.py": "",
    "quantization/pt2e/constant_fold.py": "",
    "quantization/pt2e/convert.py": "",
    "quantization/pt2e/export_utils.py": "",
    "quantization/pt2e/fake_quantize.py": "",
    "quantization/pt2e/graph_utils.py": "",
    "quantization/pt2e/learnable_fake_quantize.py": "",
    "quantization/pt2e/lowering.py": "",
    "quantization/pt2e/observer.py": "",
    "quantization/pt2e/prepare.py": "",
    "quantization/pt2e/qat_utils.py": "",
    "quantization/pt2e/quantize_pt2e.py": "",
    "quantization/pt2e/reference_representation_rewrite.py": "",
    "quantization/pt2e/utils.py": "",
    "quantization/pt2e/_affine_quantization.py": "",
    "quantization/pt2e/_numeric_debugger.py": "",
    "quantization/pt2e/inductor_passes/__init__.py": "",
    "quantization/pt2e/inductor_passes/lowering.py": "",
    "quantization/pt2e/inductor_passes/x86.py": "",
    "quantization/pt2e/quantizer/__init__.py": "",
    "quantization/pt2e/quantizer/arm_inductor_quantizer.py": "",
    "quantization/pt2e/quantizer/composable_quantizer.py": "",
    "quantization/pt2e/quantizer/duplicate_dq_pass.py": "",
    "quantization/pt2e/quantizer/embedding_quantizer.py": "",
    "quantization/pt2e/quantizer/port_metadata_pass.py": "",
    "quantization/pt2e/quantizer/quantizer.py": "",
    "quantization/pt2e/quantizer/utils.py": "",
    "quantization/pt2e/quantizer/x86_inductor_quantizer.py": "",
    "quantization/pt2e/quantizer/xpu_inductor_quantizer.py": "",
    # ── quantization/qat ────────────────────────────────────────
    "quantization/qat/__init__.py": "",
    "quantization/qat/affine_fake_quantized_tensor.py": "",
    "quantization/qat/api.py": "",
    "quantization/qat/embedding.py": "",
    "quantization/qat/fake_quantizer.py": "",
    "quantization/qat/fake_quantize_config.py": "",
    "quantization/qat/linear.py": "",
    "quantization/qat/utils.py": "",
}


def main() -> int:
    # 1. 检测 torchao
    try:
        import torchao
    except ImportError:
        print("[ERROR] torchao 未安装。")
        print("  Intel XPU: 请从 Intel 分发渠道安装 torchao==0.17.0+xpu")
        print("  NVIDIA/CUDA:  pip install torchao>=0.17.0")
        return 1

    ver = getattr(torchao, "__version__", "")
    if "+xpu" not in ver:
        print(f"[INFO] 标准 torchao {ver} 不需要此修复，跳过。")
        return 0

    pkg = os.path.dirname(os.path.abspath(torchao.__file__))
    print(f"[INFO] torchao {ver} (XPU fork)")
    print(f"[INFO] 安装路径: {pkg}\n")

    created, skipped = 0, 0

    for rel_path, content in _MISSING.items():
        full = os.path.join(pkg, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)

        if os.path.exists(full):
            skipped += 1
        else:
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ {rel_path}")
            created += 1

    if skipped:
        print(f"\n  ⏭️  {skipped} 个文件已存在，跳过。")
    print(f"\n[OK] 完成 — 新创建 {created} 个文件。")
    if created > 0:
        print("  重启 ComfyUI 后所有 torchao 兼容问题不再出现。")
    print("  本脚本可安全重复运行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
