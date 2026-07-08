"""
tint4_lora_loader.py — TINT4 LoRA Loader v8.1.0

Single LoRA injection: quant layers → _tint4_lora_entries (CPU tensors),
non-quant layers → bake-in delta with _orig_weight recovery.

Entry point resets ALL LoRA state, then rebuilds for this LoRA.

v8.1.0: cross-platform device via _get_accelerator_device().
"""
import logging
import torch
import folder_paths
import comfy.utils
from .tint4_lora_common import (
    _normalize_layer_path,
    _auto_detect_format,
    _convert_bfl_to_standard,
    _tint4_reset_all_loras,
    _rot_quarot,
    _get_candidates,
    _get_accelerator_device,
)
from .tint4_loader import TINT4Linear

log = logging.getLogger("TINT4-LoRA")


class TINT4LoRALoader:
    NAME = "TINT4 LoRA Loader"
    CATEGORY = "TINT4"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "From TINT4ModelLoader"}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0,
                    "step": 0.01,
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **kw):
        import random
        return random.random()

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"

    def load_lora(self, model, lora_name, strength):
        # ── Full reset at entry ────────────────────────────────
        _tint4_reset_all_loras(model)

        if abs(strength) < 1e-5:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise FileNotFoundError(
                f"[TINT4 LoRA] '{lora_name}' not found")

        log.info(f"[TINT4 LoRA] {lora_name} (strength={strength})")

        dm = model.model.diffusion_model
        while hasattr(dm, '_orig_mod'):
            dm = dm._orig_mod

        # ── Load & parse LoRA ──────────────────────────────────
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        if _auto_detect_format(lora_sd) == "bfl":
            lora_sd = _convert_bfl_to_standard(lora_sd)

        dev = _get_accelerator_device()  # ★ v8.1.0
        cpu = torch.device("cpu")

        lora_data: dict[str, dict] = {}
        for key, tensor in lora_sd.items():
            if "lokr_w1" in key:
                idx = key.index("lokr_w1")
                lp = _normalize_layer_path(key[:idx].rstrip("."))
                if lp:
                    lora_data.setdefault(lp, {})["lokr_w1"] = tensor
            elif "lokr_w2" in key:
                idx = key.index("lokr_w2")
                lp = _normalize_layer_path(key[:idx].rstrip("."))
                if lp:
                    lora_data.setdefault(lp, {})["lokr_w2"] = tensor
            elif "lora_up" in key or "lora_B" in key:
                idx = (key.index("lora_up") if "lora_up" in key
                       else key.index("lora_B"))
                lp = _normalize_layer_path(key[:idx].rstrip("."))
                if lp:
                    lora_data.setdefault(lp, {})["up"] = tensor
            elif "lora_down" in key or "lora_A" in key:
                idx = (key.index("lora_down") if "lora_down" in key
                       else key.index("lora_A"))
                lp = _normalize_layer_path(key[:idx].rstrip("."))
                if lp:
                    lora_data.setdefault(lp, {})["down"] = tensor
            elif key.endswith(".alpha"):
                lp = _normalize_layer_path(key[:-6])
                if lp:
                    t = tensor
                    lora_data.setdefault(lp, {})["alpha"] = (
                        float(t.mean()) if t.numel() > 1 else t.item()
                    )

        aq, ab = 0, 0
        for mod_name, module in dm.named_modules():
            norm = _normalize_layer_path(mod_name)
            if norm is None:
                continue
            is_quant = isinstance(module, TINT4Linear)

            for info, sl, se in _get_candidates(
                    norm, lora_data, module, is_quant):

                # ── LoKr ───────────────────────────────────────
                if "lokr_w1" in info and "lokr_w2" in info:
                    if not is_quant:
                        continue
                    w1 = info["lokr_w1"].to(cpu, torch.float16).clone()
                    w2 = info["lokr_w2"].to(cpu, torch.float16).clone()
                    mult = (info.get("alpha", w1.shape[0])
                            / max(w1.shape[0], 1)) * strength
                    _rot_quarot(module, w2, dev)
                    if sl is not None and w1.shape[0] != (se - sl):
                        continue
                    le = getattr(module, '_tint4_lora_entries', None)
                    if le is None:
                        le = {}
                        object.__setattr__(module, '_tint4_lora_entries', le)
                    le.pop(lora_name, None)
                    e = ("lokr", w1, w2, mult, w1.shape[0])
                    if sl is not None:
                        e = ("lokr", w1, w2, mult, w1.shape[0], sl, se)
                    le[lora_name] = [e]
                    aq += 1
                    continue

                # ── Standard LoRA ──────────────────────────────
                up, down = info.get("up"), info.get("down")
                if up is None or down is None:
                    continue
                mult = (info.get("alpha", up.shape[1])
                        / max(up.shape[1], 1)) * strength
                if sl is not None and up.shape[0] != (se - sl):
                    continue

                if not is_quant:
                    A = down.to(cpu, torch.float16).clone()
                    B = up.to(cpu, torch.float16).clone()
                    A = _rot_quarot(module, A, dev)
                    delta = (B @ A).mul_(mult)
                    if not hasattr(module, 'weight') or module.weight is None:
                        continue
                    bs = getattr(module, '_tint4_bake_state', None)
                    if bs is None:
                        bs = {'_orig_weight': module.weight.data.clone()}
                        object.__setattr__(module, '_tint4_bake_state', bs)
                    bs[lora_name] = {
                        'delta': delta.clone(),
                        'sl': sl, 'se': se,
                    }
                    d = delta.to(device=module.weight.device)
                    if sl is not None:
                        module.weight.data[sl:se].add_(d)
                    else:
                        module.weight.data.add_(d)
                    ab += 1
                    continue

                A = down.to(cpu, torch.float16).clone()
                B = up.to(cpu, torch.float16).clone()
                A = _rot_quarot(module, A, dev)
                le = getattr(module, '_tint4_lora_entries', None)
                if le is None:
                    le = {}
                    object.__setattr__(module, '_tint4_lora_entries', le)
                le.pop(lora_name, None)
                e = (A, B, mult) if sl is None else (A, B, mult, sl, se)
                le[lora_name] = [e]
                aq += 1

        del lora_sd, lora_data

        # ── Track active LoRAs ─────────────────────────────────
        if not hasattr(model.model, '_tint4_loras'):
            object.__setattr__(model.model, '_tint4_loras', [])
        model.model._tint4_loras.append({
            "name": lora_name, "strength": strength, "path": lora_path,
        })

        log.info(f"[TINT4 LoRA] {lora_name} → {aq} quant + {ab} bake-in")
        return (model,)


NODE_CLASS_MAPPINGS = {"TINT4LoRALoader": TINT4LoRALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TINT4LoRALoader": "TINT4 LoRA Loader"}
