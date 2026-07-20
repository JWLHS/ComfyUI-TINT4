"""
tint4_loader_wan.py — TINT4 WAN Video Model Loader v1.2

Standalone loader for WAN INT4 models with built-in block swap.
Only <blocks_to_keep> transformer blocks on GPU at once.

  - _qt NOT cached — rebuilt from CPU qdata/scale/zp each forward,
    then immediately deleted (same strategy as verified LTX loader).
  - Empty-cache every 100 layers.
  - _vgpu (LoRA GPU cache) cached — first upload, then reuse.
  - _apply() delegates to super() only — quantised data stays on CPU.
  - Placeholder weight deleted at injection time.
  - Dual key lookup (bare + model.diffusion_model. prefix) for
    animate and i2v model compatibility.
  - Auto-detect QuaRot: if marker=0 but gs!=128, force ON.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc, os, json, hashlib, math, logging
import folder_paths
import comfy.sd
import comfy.model_detection
import comfy.model_management as mm
import comfy.utils
import comfy.ops
from torchao.quantization.quantize_.workflows.int4.int4_plain_int32_tensor import (
    Int4PlainInt32Tensor,
)
from safetensors import safe_open
from .tint4_loader import TINT4Linear

log = logging.getLogger("TINT4-WAN-Loader")
_orig_detect = comfy.model_detection.detect_unet_config


class TINT4LinearWAN(TINT4Linear):

    def forward(self, x):
        x2 = x.reshape(-1, x.shape[-1])
        if self._use_quarot and self._hadamard_H is not None:
            try:
                from .wint8_quarot import rotate_activation
                x2 = rotate_activation(x2, self._hadamard_H, self._group_size)
            except Exception:
                pass
        dev = x.device

        qd = self._qdata.to(dev, non_blocking=True)
        sc = self._scale.to(dev, non_blocking=True)
        zp = self._zp.to(dev, non_blocking=True)
        qt = Int4PlainInt32Tensor(
            qd, sc, zp,
            self._block_size, [self.out_features, self.in_features],
        )
        out = F.linear(x2, qt, None)
        del qd, sc, zp, qt

        n = getattr(self, '_fwd_n', 0) + 1
        object.__setattr__(self, '_fwd_n', n)
        if n % 100 == 0:
            try:
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
            except Exception:
                pass

        entries = self._tint4_lora_entries
        if entries and len(entries) > 0:
            cd = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
            x2c = x2.to(cd) if x2.dtype != cd else x2
            gpu = getattr(self, '_vgpu', None)
            if gpu is None:
                gpu = {}
                object.__setattr__(self, '_vgpu', gpu)
            for lora_entries in entries.values():
                for e in lora_entries:
                    etype = e[0]
                    if isinstance(etype, str) and etype == "delta":
                        _, delta_cpu, mult = e[:3]
                        sl, se = (e[3], e[4]) if len(e) > 4 else (None, None)
                        dk = id(delta_cpu)
                        if dk not in gpu:
                            gpu[dk] = delta_cpu.to(device=dev, dtype=cd)
                        lo = x2c @ gpu[dk].T * mult
                        if sl is not None:
                            out[:, sl:se] += lo
                        else:
                            out += lo
                        del lo; continue
                    if isinstance(etype, str) and etype == "lokr":
                        _, w1, w2, mult, factor = e[:5]
                        sl, se = (e[5], e[6]) if len(e) > 6 else (None, None)
                        of2, if2 = w2.shape
                        k1, k2 = id(w1), id(w2)
                        if k1 not in gpu:
                            gpu[k1] = w1.to(device=dev, dtype=cd)
                        if k2 not in gpu:
                            gpu[k2] = w2.to(device=dev, dtype=cd)
                        w1x = gpu[k1].repeat_interleave(of2 // factor, dim=0) \
                                        .repeat_interleave(if2 // factor, dim=1)
                        dw = (w1x * gpu[k2]).mul_(mult)
                        lo = x2c @ dw.T
                        if sl is not None:
                            if lo.shape[1] == (se - sl):
                                out[:, sl:se] += lo
                        elif lo.shape == out.shape:
                            out += lo
                        del lo, dw, w1x; continue
                    A, B, mult = e[:3]
                    sl, se = (e[3], e[4]) if len(e) > 4 else (None, None)
                    ka, kb = id(A), id(B)
                    if ka not in gpu:
                        gpu[ka] = A.to(device=dev, dtype=cd)
                    if kb not in gpu:
                        gpu[kb] = B.to(device=dev, dtype=cd)
                    lo = (x2c @ gpu[ka].T) @ gpu[kb].T * mult
                    if sl is not None:
                        if lo.shape[1] == (se - sl):
                            out[:, sl:se] += lo
                    elif lo.shape[1] == out.shape[1]:
                        out += lo
                    del lo
            if x2c is not x2:
                del x2c

        if self.bias is not None:
            out += self.bias.to(device=dev, dtype=out.dtype)
        return out.reshape(*x.shape[:-1], out.shape[-1])

    def _apply(self, fn, *args, **kwargs):
        return super()._apply(fn, *args, **kwargs)

    def _flush_vgpu(self):
        gpu = getattr(self, '_vgpu', None)
        if gpu is not None:
            for v in gpu.values():
                del v
            gpu.clear()
            object.__setattr__(self, '_vgpu', None)

    def release_xpu(self):
        super().release_xpu()
        self._flush_vgpu()
        try:
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        except Exception:
            pass


def _flush_all_vgpu(dm):
    for m in dm.modules():
        if isinstance(m, TINT4LinearWAN):
            m._flush_vgpu()


def _normalize_index_path(name):
    for pf in ["diffusion_model.", "model.diffusion_model.", "model."]:
        if name.startswith(pf):
            name = name[len(pf):]; break
    if name.startswith("img_in") or name.startswith("final_layer"):
        return None
    for old, new in [
        ("layers.", "blocks."), ("joint_blocks.", "blocks."),
        ("transformer_blocks.", "blocks."), ("double_blocks.", "blocks."),
        ("single_blocks.", "blocks."),
    ]:
        if name.startswith(old):
            name = new + name[len(old):]; break
    name = name.replace(".ff.", ".mlp.").replace(".feed_forward.", ".mlp.")
    name = name.replace(".img_attn.", ".attn.").replace(".txt_attn.", ".attn.")
    name = name.replace(".attention.", ".attn.")
    for a, b in [(".to_q", ".wq"), (".to_k", ".wk"), (".to_v", ".wv"),
                 (".to_out.0", ".wo"), (".to_out", ".wo"), (".to_gate", ".gate"),
                 (".q_proj", ".wq"), (".k_proj", ".wk"), (".v_proj", ".wv"),
                 (".out_proj", ".wo")]:
        name = name.replace(a, b)
    name = name.replace(".self_attn.q", ".attn.wq")
    name = name.replace(".self_attn.k", ".attn.wk")
    name = name.replace(".self_attn.v", ".attn.wv")
    name = name.replace(".self_attn.o", ".attn.wo")
    name = name.replace(".attn.out", ".attn.wo")
    return name


def _build_lora_index(dm):
    index = {}
    for name, module in dm.named_modules():
        if not isinstance(module, (TINT4Linear, nn.Linear)):
            continue
        norm = _normalize_index_path(name)
        if norm is None:
            continue
        if norm.endswith(".attn.qkv") and isinstance(module, TINT4Linear):
            out_f = module.out_features
            if out_f % 3 == 0:
                hs = out_f // 3
                base = norm.rsplit(".attn.qkv", 1)[0]
                index[f"{base}.attn.wq"] = module
                index[f"{base}.attn.wk"] = module
                index[f"{base}.attn.wv"] = module
        elif norm.endswith(".attn.qkv"):
            w = getattr(module, 'weight', None)
            out_f = w.shape[0] if w is not None and hasattr(w, 'shape') else 0
            if out_f > 0 and out_f % 3 == 0:
                hs = out_f // 3
                base = norm.rsplit(".attn.qkv", 1)[0]
                index[f"{base}.attn.wq"] = module
                index[f"{base}.attn.wk"] = module
                index[f"{base}.attn.wv"] = module
        index[norm] = module
    return index


def _detect_wan(state_dict, key_prefix):
    keys = list(state_dict.keys())
    pe_key = f"{key_prefix}patch_embedding.weight"
    if pe_key not in keys:
        return None
    pe = state_dict[pe_key]
    if pe.ndim != 5 or pe.shape[1] != 16:
        return None
    ffn_key = f"{key_prefix}blocks.0.ffn.0.weight"
    if ffn_key not in keys:
        return None
    dim = int(pe.shape[0])
    ffn_dim = int(state_dict[ffn_key].shape[0])
    in_dim = int(pe.shape[1])
    num_layers = comfy.model_detection.count_blocks(keys, f"{key_prefix}blocks." + "{}.")
    dit_config = {
        "image_model": "wan2.1", "dim": dim, "out_dim": 16,
        "num_heads": dim // 128, "ffn_dim": ffn_dim,
        "num_layers": num_layers, "patch_size": (1, 2, 2),
        "freq_dim": 256, "window_size": (-1, -1),
        "qk_norm": True, "cross_attn_norm": True,
        "eps": 1e-6, "in_dim": in_dim,
    }
    dit_config["model_type"] = (
        "i2v" if f"{key_prefix}img_emb.proj.0.bias" in keys else "t2v")
    return dit_config


_QUANT_META_SUFFIXES = (
    ".weight_scale", ".weight_zp", ".weight_b0", ".weight_b1",
    ".weight_sh0", ".weight_sh1", ".comfy_quant",
)


def _extract_int4_from_sd(sd):
    quant = {}
    specs = []
    for k in list(sd.keys()):
        if not k.endswith(".weight"):
            continue
        v = sd[k]
        if v.dtype not in (torch.int32, torch.int8, torch.uint8):
            continue
        base = k.rsplit(".weight", 1)[0]
        scale_key = f"{base}.weight_scale"
        zp_key = f"{base}.weight_zp"
        b0_key = f"{base}.weight_b0"
        b1_key = f"{base}.weight_b1"
        if scale_key not in sd or zp_key not in sd:
            continue
        b0_t = sd.get(b0_key)
        b1_t = sd.get(b1_key)
        if b0_t is None or b1_t is None:
            continue
        out_f = v.shape[0]
        in_f = v.shape[1] * 8 if v.dtype == torch.int32 else v.shape[1] * 2
        quant[base] = {
            "qdata": v, "scale": sd[scale_key], "zp": sd[zp_key],
            "b0": b0_t.item(), "b1": b1_t.item(),
            "out_f": out_f, "in_f": in_f,
        }
        specs.append((base, out_f, in_f))
        del sd[k], sd[scale_key], sd[zp_key]
        if b0_key in sd:
            del sd[b0_key]
        if b1_key in sd:
            del sd[b1_key]
    specs.sort(key=lambda x: x[0])
    chosen, seen = [], set()
    for base, out_f, in_f in specs:
        pfx = ".".join(base.split(".")[:3])
        if pfx not in seen:
            seen.add(pfx)
            chosen.append((base, out_f, in_f))
        if len(chosen) >= 20:
            break
    for base, out_f, in_f in chosen:
        sd[f"{base}.weight"] = torch.zeros(out_f, in_f, dtype=torch.float16)
    return quant


class TINT4WANLoader:

    NAME = "TINT4 WAN Loader"
    CATEGORY = "TINT4"
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("diffusion_models")
        if not files:
            files = ["none"]
        return {
            "required": {
                "unet_name": (files, {"tooltip": "INT4 WAN model"}),
                "blocks_to_keep": ("INT", {
                    "default": 2, "min": 1, "max": 40,
                    "tooltip": "Blocks kept on GPU.  Lower = less VRAM.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = "Load WAN INT4 model with built-in block swap."

    def load_model(self, unet_name, blocks_to_keep):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise FileNotFoundError(f"'{unet_name}' not found")
        log.info(f"[TINT4 WAN] Loading: {unet_name}")

        sd = comfy.utils.load_torch_file(unet_path, safe_load=True)
        sd_metadata = {}
        with safe_open(unet_path, framework="pt") as f:
            sd_metadata = f.metadata() or {}

        is_quarot = False
        quarot_gs = 128
        for k in list(sd.keys()):
            if k == "__tint4_quarot__":
                is_quarot = (sd.pop(k).item() == 1)
            elif k == "__tint4_group_size__":
                quarot_gs = sd.pop(k).item()
            elif k in ("__tint4_format__", "__tint4_model_type__"):
                sd.pop(k, None)

        # Fix: quantizer may omit/zero the QuaRot marker.
        # If gs != default (128), rotation was applied.
        if not is_quarot and quarot_gs != 128:
            log.warning("[TINT4 WAN] QuaRot marker=0 but gs=%d — forcing ON", quarot_gs)
            is_quarot = True

        quant = _extract_int4_from_sd(sd)

        for k in list(sd.keys()):
            for s in _QUANT_META_SUFFIXES:
                if k.endswith(s):
                    del sd[k]
                    break

        log.info(f"[TINT4 WAN] QuaRot={'ON' if is_quarot else 'OFF'}  "
                 f"gs={quarot_gs}  {len(quant)} INT4 layers, {len(sd)} sd keys")

        def _detect_wrapper(state_dict, key_prefix, metadata=None):
            for _ in range(5):
                try:
                    result = _orig_detect(state_dict, key_prefix, metadata)
                    if result is not None:
                        return result
                    return _detect_wan(state_dict, key_prefix)
                except (KeyError, IndexError) as e:
                    missing = e.args[0] if isinstance(e, KeyError) and e.args else None
                    if missing and missing in state_dict:
                        del state_dict[missing]
                    if missing and missing.endswith(".weight"):
                        for pf in ["model.diffusion_model.", "diffusion_model.", "model."]:
                            if missing.startswith(pf):
                                base = missing[len(pf):].rsplit(".weight", 1)[0]
                                break
                        else:
                            base = missing.rsplit(".weight", 1)[0]
                        q = quant.get(f"model.diffusion_model.{base}") or quant.get(base)
                        if q:
                            state_dict[missing] = torch.zeros(q["out_f"], q["in_f"],
                                                              dtype=torch.float16)
                            continue
                    if missing and missing not in state_dict:
                        state_dict[missing] = torch.zeros(1, 1, dtype=torch.float16)
                        continue
                    raise
            return None

        comfy.model_detection.detect_unet_config = _detect_wrapper
        try:
            model = comfy.sd.load_diffusion_model_state_dict(
                sd, model_options={"custom_operations": comfy.ops.manual_cast},
                metadata=sd_metadata)
        finally:
            comfy.model_detection.detect_unet_config = _orig_detect

        if model is None:
            raise RuntimeError(
                f"[TINT4 WAN] Model detection failed for '{unet_name}'. "
                f"Check that the file is a valid WAN model."
            )

        dm = model.model.diffusion_model
        while hasattr(dm, '_orig_mod'):
            dm = dm._orig_mod
        del sd; gc.collect()

        count = 0
        for name, module in dm.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            base = name
            for pf in ["model.diffusion_model.", "diffusion_model.", "model."]:
                if base.startswith(pf):
                    base = base[len(pf):]; break

            qinfo = quant.get(base)
            if qinfo is None:
                qinfo = quant.get(f"model.diffusion_model.{base}")
            if qinfo is None:
                continue

            bias = module.bias.data.clone() if module.bias is not None else None
            new_mod = TINT4LinearWAN(
                qinfo["in_f"], qinfo["out_f"],
                qinfo["qdata"], qinfo["scale"], qinfo["zp"],
                [qinfo["b0"], qinfo["b1"]], bias,
            )
            parent = dm
            path_parts = name.split(".")
            for p in path_parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            leaf = path_parts[-1]
            if leaf.isdigit():
                parent[int(leaf)] = new_mod
            else:
                setattr(parent, leaf, new_mod)

            if 'weight' in new_mod._parameters:
                del new_mod._parameters['weight']

            count += 1
        log.info(f"[TINT4 WAN] Injected {count} TINT4LinearWAN layers")

        if is_quarot:
            from .wint8_quarot import build_hadamard
            H = build_hadamard(quarot_gs, device="cpu", dtype=torch.float32)
            for m in dm.modules():
                if isinstance(m, TINT4LinearWAN):
                    object.__setattr__(m, '_use_quarot', True)
                    object.__setattr__(m, '_group_size', quarot_gs)
                    object.__setattr__(m, '_hadamard_H', H)

        object.__setattr__(dm, '_tint4_quarot_enabled', is_quarot)
        object.__setattr__(dm, '_tint4_group_size', quarot_gs)
        index = _build_lora_index(dm)
        object.__setattr__(dm, '_tint4_lora_index', index)
        log.info(f"[TINT4 WAN] LoRA index: {len(index)} entries "
                 f"(QuaRot={'ON' if is_quarot else 'OFF'})")

        # ── Built-in block swap ──────────────────────────────
        blocks = list(dm.blocks)
        n_blocks = len(blocks)
        keep = min(blocks_to_keep, n_blocks)
        main_dev = mm.get_torch_device()
        offload_dev = model.offload_device

        log.info("[TINT4 WAN] Block swap: %d blocks | keep=%d | offload=%s",
                 n_blocks, keep, offload_dev)

        for i, blk in enumerate(blocks):
            blk.to(main_dev if i < keep else offload_dev)

        all_hooks = []

        for i, blk in enumerate(blocks):

            def _make_pre_hook(_blk):
                def pre_hook(_module, _inputs):
                    _blk.to(main_dev)
                return pre_hook

            def _make_post_hook(idx):
                def post_hook(_module, _inputs, _output):
                    off_idx = idx - keep
                    if off_idx >= 0:
                        blocks[off_idx].to(offload_dev)
                        try:
                            torch.xpu.synchronize()
                            torch.xpu.empty_cache()
                        except Exception:
                            pass
                return post_hook

            h_pre = blk.register_forward_pre_hook(_make_pre_hook(blk))
            h_post = blk.register_forward_hook(_make_post_hook(i))
            all_hooks.append(h_pre)
            all_hooks.append(h_post)

        try:
            from .tint4_aimdo import patch_model_for_aimdo
            patch_model_for_aimdo(model)
        except Exception:
            pass

        _orig_detach = model.detach

        def _wan_detach(unpatch_all=True):
            for h in all_hooks:
                h.remove()
            all_hooks.clear()
            for blk in blocks:
                blk.to(offload_dev)
            _dm = model.model.diffusion_model
            while hasattr(_dm, '_orig_mod'):
                _dm = _dm._orig_mod
            _flush_all_vgpu(_dm)
            for m in _dm.modules():
                if isinstance(m, TINT4LinearWAN):
                    m.release_xpu()
            gc.collect()
            try:
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
            except Exception:
                pass
            return _orig_detach(unpatch_all)

        object.__setattr__(model, 'detach', _wan_detach)
        log.info(f"[TINT4 WAN] Loaded '{unet_name}'")
        return (model,)


NODE_CLASS_MAPPINGS = {"TINT4WANLoader": TINT4WANLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TINT4WANLoader": "TINT4 WAN Loader"}
