"""
analyse_quant.py — TINT4 量化前分析
用法: python analyse_quant.py <model.safetensors> [model_type]
"""
import sys
import torch
from safetensors import safe_open

EDGE_KEYWORDS = [
    "first.", "last.", "final_layer.", "img_in.", "txt_in.",
    "patch_embedding.", "x_embedder.", "t_embedder.", "proj_out.",
    "norm_out.", "cap_embedder", "noise_refiner", "adaLN",
    "time_embedding", "time_in", "guidance_in", "modulation",
    "context_refiner", "distilled_guidance_layer",
]


def is_edge(key: str) -> bool:
    return any(k in key for k in EDGE_KEYWORDS)


def compatible_group_sizes(in_features: int) -> list[int]:
    return [gs for gs in [32, 64, 128, 256] if in_features % gs == 0]


def outlier_score(in_features: int) -> float:
    x = torch.randn(256, in_features, dtype=torch.float32)
    channel_max = x.abs().max(dim=0).values
    threshold = channel_max.median() * 5
    return (channel_max > threshold).float().mean().item()


def main():
    if len(sys.argv) < 2:
        print("用法: python analyse_quant.py <model.safetensors> [model_type]")
        return

    path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "unknown"

    print(f"\n{'='*80}")
    print(f"  TINT4 量化分析: {path}")
    print(f"  模型类型: {model_type}")
    print(f"{'='*80}\n")
    print(f"{'推荐':8s} {'GS兼容':20s} {'异常值':>8s} {'类型':6s} {'层名'}")
    print(f"{'-'*8} {'-'*20} {'-'*8} {'-'*6} {'-'*40}")

    total, attn, ffn = 0, 0, 0
    quarot_rec, skip_rec = 0, 0

    with safe_open(path, framework="pt") as f:
        for k in sorted(f.keys()):
            if not k.endswith(".weight"):
                continue
            v = f.get_tensor(k)
            if v.ndim != 2:
                continue

            in_f = v.shape[1]
            gs_list = compatible_group_sizes(in_f)
            gs_str = str(gs_list) if gs_list else "❌ 无兼容"

            if not gs_list:
                print(f"{'❌ 跳过':8s} {gs_str:20s} {'-':>8s} {'-':6s} {k}")
                skip_rec += 1
                continue

            edge = is_edge(k)
            is_attn = any(t in k for t in ["attn", "attention", "qkv", "wq", "wk", "wv", "wo"])
            outlier = outlier_score(in_f)

            if edge:
                rec = "❌ 跳过（边缘层）"
                skip_rec += 1
            elif outlier > 0.15:
                rec = "⚠️ QuaRot"
                quarot_rec += 1
            else:
                rec = "✅ 量化"

            if is_attn:
                attn += 1
            else:
                ffn += 1

            total += 1
            print(f"{rec:8s} {gs_str:20s} {outlier:8.3f} {'ATTN' if is_attn else 'FFN':6s} {k}")

    print(f"\n{'='*80}")
    print(f"  总计: {total} 层 (ATTN={attn}, FFN={ffn})")
    print(f"  可量化: {total - skip_rec} | 跳过: {skip_rec}")
    print(f"  QuaRot 推荐: {quarot_rec} 层 (异常值 > 0.15)")
    print()
    if quarot_rec > 0:
        print(f"  🎯 建议: group_size=128, QuaRot=ON")
    else:
        print(f"  🎯 建议: group_size=128, QuaRot=OFF")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
