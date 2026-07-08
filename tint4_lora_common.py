"""
tint4_lora_common.py
────────────────────
Shared LoRA utilities — key normalization, format detection,
full-reset helper, QuaRot rotation for LoRA, QKV split.
No global registry.
"""
import torch


def _get_accelerator_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _empty_accelerator_cache():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def _normalize_layer_path(path: str) -> str | None:
    stripped_prefix = None
    for pf in ["lora_transformer_", "lora_unet_", "lycoris_"]:
        if path.startswith(pf):
            path = path[len(pf):].replace("_", ".")
            stripped_prefix = pf
            break
    if stripped_prefix is None:
        if path.startswith("transformer."):
            path = path[len("transformer."):]
        elif path.startswith("diffusion_model."):
            path = path[len("diffusion_model."):]
    if path.startswith("img_in") or path.startswith("final_layer"):
        return None
    if path.startswith("text_fusion.layerwise_blocks."):
        path = "blocks." + path[len("text_fusion.layerwise_blocks."):]
    for old, new in [
        ("layers.", "blocks."), ("joint_blocks.", "blocks."),
        ("transformer_blocks.", "blocks."), ("double_blocks.", "blocks."),
        ("single_blocks.", "blocks."),
    ]:
        if path.startswith(old):
            path = new + path[len(old):]; break
    path = path.replace(".ff.", ".mlp.").replace(".feed_forward.", ".mlp.")
    path = path.replace(".img_attn.", ".attn.").replace(".txt_attn.", ".attn.")
    path = path.replace(".attention.", ".attn.")
    path = path.replace(".to_q", ".wq").replace(".to_k", ".wk")
    path = path.replace(".to_v", ".wv").replace(".to_out.0", ".wo")
    path = path.replace(".to_out", ".wo").replace(".to_gate", ".gate")
    path = path.replace(".q_proj", ".wq").replace(".k_proj", ".wk")
    path = path.replace(".v_proj", ".wv").replace(".out_proj", ".wo")
    path = path.replace(".self_attn.q", ".attn.wq")
    path = path.replace(".self_attn.k", ".attn.wk")
    path = path.replace(".self_attn.v", ".attn.wv")
    path = path.replace(".self_attn.o", ".attn.wo")
    path = path.replace(".attn.out", ".attn.wo")
    return f"diffusion_model.{path}"


def _auto_detect_format(sd: dict) -> str:
    for key in sd:
        if "single_blocks" in key or "double_blocks" in key:
            return "bfl"
        if "diffusion_model.blocks" in key or "diffusion_model.layers" in key:
            return "standard"
    return "unknown"


def _convert_bfl_to_standard(sd: dict) -> dict:
    out = {}
    for key, tensor in sd.items():
        if "qkv.lora" in key or "proj.lora" in key or "ff.lora" in key:
            for prefix in ["double_blocks", "single_blocks"]:
                if key.startswith(prefix):
                    break
            else:
                out[key] = tensor; continue
            rest = key[len(prefix) + 1:]
            parts = rest.split(".")
            block_num = parts[0]
            attn_type = parts[1] if len(parts) > 1 and "attn" in parts[1] else "attn"
            if "lora_B" in key: lora_type = "up"
            elif "lora_A" in key: lora_type = "down"
            elif "lora_up" in key: lora_type = "up"
            elif "lora_down" in key: lora_type = "down"
            else: out[key] = tensor; continue
            stem = "qkv" if "qkv" in key else "proj"
            std_key = f"diffusion_model.blocks.{block_num}.{attn_type}.{stem}"
            out[f"{std_key}.lora_{lora_type}.weight"] = tensor
        else:
            out[key] = tensor
    return out


def _tint4_reset_all_loras(model) -> None:
    from .tint4_loader import TINT4Linear

    dm = model.model.diffusion_model
    while hasattr(dm, '_orig_mod'):
        dm = dm._orig_mod

    for m in dm.modules():
        if hasattr(m, '_tint4_lora_entries'):
            object.__setattr__(m, '_tint4_lora_entries', None)

        bs = getattr(m, '_tint4_bake_state', None)
        if bs is not None and '_orig_weight' in bs:
            if isinstance(m, TINT4Linear):
                object.__setattr__(m, '_tint4_bake_state', None)
            elif hasattr(m, 'weight') and m.weight is not None:
                orig = bs['_orig_weight']
                try:
                    m.weight.data.copy_(orig.to(
                        device=m.weight.device, dtype=m.weight.dtype))
                except Exception:
                    pass
            object.__setattr__(m, '_tint4_bake_state', None)
        elif bs is not None:
            object.__setattr__(m, '_tint4_bake_state', None)

    object.__setattr__(model.model, '_tint4_loras', [])
    object.__setattr__(model.model, '_lora_needs_reset', False)


def _rot_quarot(module, tensor: torch.Tensor, dev: torch.device) -> torch.Tensor:
    if getattr(module, '_use_quarot', False):
        H = getattr(module, '_hadamard_H', None)
        gs = getattr(module, '_group_size', 128)
        if H is not None and gs > 0 and tensor.shape[1] % gs == 0:
            Hd = H.to(tensor.device, dtype=torch.float16)
            ng = tensor.shape[1] // gs
            return (tensor.reshape(tensor.shape[0], ng, gs) @ Hd.T).reshape(
                tensor.shape[0], tensor.shape[1])
    return tensor


def _get_candidates(norm: str, lora_data: dict, module, is_quant: bool) -> list:
    cands = []
    if norm.endswith(".attn.qkv"):
        out_f = (module.out_features if is_quant
                 else (module.weight.shape[0] if hasattr(module, 'weight') else 0))
        if out_f > 0:
            hs = out_f // 3
            if hs * 3 == out_f:
                for suf, st, en in [(".attn.wq", 0, hs),
                                    (".attn.wk", hs, 2 * hs),
                                    (".attn.wv", 2 * hs, 3 * hs)]:
                    info = lora_data.get(norm.replace(".attn.qkv", suf))
                    if info:
                        cands.append((info, st, en))
    info = lora_data.get(norm)
    if info:
        cands.append((info, None, None))
    return cands
