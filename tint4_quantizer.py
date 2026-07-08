"""
tint4_quantizer.py — TINT4 Model Quantizer  v8.0

Offline: fp16/bf16/int8 → optional QuaRot rotation → torchao INT4 → safetensors

Output format (v7 — weight=int32):
  {base}.weight        — int32 qdata  (out_f, in_f//8)
  {base}.weight_scale  — fp16  (blocks, out_f)
  {base}.weight_zp     — int8  (blocks, out_f)
  {base}.weight_b0     — int32 block_size[0]
  {base}.weight_b1     — int32 block_size[1]
  {base}.comfy_quant   — JSON metadata (quarot / group_size / format)

Global markers:
  __tint4_format__      — uint8 (1)
  __tint4_quarot__      — uint8 (0 or 1)
  __tint4_group_size__  — int32
  __tint4_model_type__  — uint8 string

Verified models: krea2, flux2.
Others: exclusion lists derived from ComfyUI 0.27.0 model_detection.py.
"""
import os
import json
import logging
import torch
import torch.nn as nn
import folder_paths
import comfy.utils
from safetensors import safe_open
from torchao.quantization import Int4WeightOnlyConfig, quantize_

log = logging.getLogger("TINT4-Quantizer")

# ── Exclusion lists ──────────────────────────────────────────────────
# Keys containing any of these substrings are kept as fp16.
# Derived from model_detection.py heuristics + empirical testing.
_EXCLUSIONS = {
    "flux2": [
        "img_in", "time_in", "guidance_in", "txt_in", "final_layer",
        "double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation",
    ],
    "z-image": [
        "cap_embedder", "t_embedder", "x_embedder", "cap_pad_token",
        "context_refiner", "final_layer", "noise_refiner", "adaLN",
        "x_pad_token", "cap_embedder.0",
        "attention_norm1", "attention_norm2",
        "ffn_norm1", "ffn_norm2", "k_norm", "q_norm",
        # v8.0: removed "layers.0." — first transformer block now quantized
    ],
    "chroma": [
        "distilled_guidance_layer", "final_layer", "img_in", "txt_in",
        "nerf_image_embedder", "nerf_blocks",
        "nerf_final_layer_conv", "__x0__",
    ],
    "wan": [
        "patch_embedding", "text_embedding", "time_embedding",
        "time_projection", "head", "img_emb", "motion_encoder",
        "modulation", "norm_q", "norm_k", "norm3",
    ],
    "ltx2": [
        "adaln_single", "audio_adaln_single",
        "audio_caption_projection", "audio_patchify_proj",
        "audio_proj_out", "audio_scale_shift_table",
        "av_ca_a2v_gate_adaln_single",
        "av_ca_audio_scale_shift_adaln_single",
        "av_ca_v2a_gate_adaln_single",
        "av_ca_video_scale_shift_adaln_single",
        "caption_projection", "patchify_proj", "proj_out",
        "scale_shift_table", "learnable_registers",
        "q_norm", "k_norm",
    ],
    "qwen": [
        "text_encoders", "time_text_embed", "img_in",
        "norm_out", "proj_out", "txt_in",
        "norm_added_k", "norm_added_q", "norm_k", "norm_q",
        "txt_norm", "transformer_blocks.0.img_mod.1",
    ],
    "ernie": [
        "time", "x_embedder", "adaLN", "final",
        "text_proj", "norm", "layers.0.", "layers.35",
    ],
    "hidream": [
        "patch_embedding", "time_text_embed", "norm_out", "proj_out",
    ],
    "boogu": [
        "embed", "refine", "norm_out",
    ],
    "krea2": [
        "first", "last", "tmlp", "tproj", "txtfusion", "txtmlp",
    ],
    "ideogram4": [
        "embed_image_indicator", "t_embedding", "proj",
    ],
    "auto": [],
}
MODEL_TYPES = list(_EXCLUSIONS.keys())


def _is_excluded(key: str, model_type: str) -> bool:
    for p in _EXCLUSIONS.get(model_type, []):
        if p in key:
            return True
    return False


def _should_quantize(key: str, tensor: torch.Tensor,
                     model_type: str) -> bool:
    if tensor.ndim != 2:
        return False
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32,
                             torch.float8_e4m3fn, torch.float8_e5m2,
                             torch.int8):
        return False
    if _is_excluded(key, model_type):
        return False
    return True


def _get_available_devices() -> list[str]:
    choices = ["cpu"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        choices.append("xpu")
    if torch.cuda.is_available():
        choices.append("cuda")
    return choices


def _str_tensor(s: str) -> torch.Tensor:
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.uint8)


def _make_comfy_quant_meta(quarot: bool = False,
                            group_size: int | None = None) -> torch.Tensor:
    payload = {"format": "tint4_torchao", "per_block": True}
    if quarot and group_size:
        payload["quarot"] = True
        payload["group_size"] = group_size
    return torch.tensor(
        list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)


class TINT4ModelQuantizer:
    NAME = "TINT4 Model Quantizer"
    CATEGORY = "TINT4"

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("diffusion_models")
        if not files:
            files = ["none"]
        devs = _get_available_devices()
        dflt = "xpu" if "xpu" in devs else (
            "cuda" if "cuda" in devs else "cpu")
        return {
            "required": {
                "model_name": (files, {
                    "tooltip": "Source model (bf16/fp16/fp8/int8)",
                }),
                "model_type": (MODEL_TYPES, {"default": "flux2"}),
                "enable_quarot": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Hadamard rotation — strongly recommended for INT4",
                }),
                "group_size": ("INT", {
                    "default": 128, "min": 32, "max": 256, "step": 32,
                }),
                "device": (devs, {"default": dflt}),
                "output_filename": ("STRING", {
                    "default": "model_tint4",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "quantize"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Quantize diffusion model to torchao INT4 (v8.0: weight=int32)")

    def quantize(self, model_name, model_type, enable_quarot, group_size,
                 device, output_filename):
        src_path = folder_paths.get_full_path(
            "diffusion_models", model_name)
        if src_path is None:
            raise FileNotFoundError(f"'{model_name}' not found")

        output_dir = folder_paths.get_output_directory()
        dst_path = os.path.join(
            output_dir, f"{output_filename}.safetensors")
        dst_dir = os.path.dirname(dst_path)
        if dst_dir and not os.path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)

        dev = torch.device(device)
        log.info(
            f"[TINT4] device={dev}  gs={group_size}"
            f"  quarot={enable_quarot}")

        sd = comfy.utils.load_torch_file(src_path, safe_load=True)
        log.info(f"[TINT4] Loaded {len(sd)} keys")

        src_metadata = {}
        try:
            with safe_open(src_path, framework="pt") as f:
                src_metadata = f.metadata() or {}
        except Exception:
            pass

        # ── Boogu: force group_size=32 ────────────────────────────────
        actual_gs = group_size
        if model_type == "boogu":
            actual_gs = 32
            log.info("[TINT4] Boogu: overriding group_size → 32")

        # ── QuaRot ────────────────────────────────────────────────────
        H = None
        if enable_quarot:
            from .wint8_quarot import build_hadamard, rotate_weight
            H = build_hadamard(
                actual_gs, device=str(dev), dtype=torch.float32)
            log.info(f"[TINT4] QuaRot enabled, gs={actual_gs}")

        quantized, excluded, skipped = 0, 0, 0

        for key in list(sd.keys()):
            tensor = sd[key]
            if not isinstance(tensor, torch.Tensor):
                continue

            base = (key.rsplit(".weight", 1)[0]
                    if key.endswith(".weight") else None)

            # FP8 → FP16 (always)
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                sd[key] = tensor.to(torch.float16)

            if base is None or not _should_quantize(
                    key, tensor, model_type):
                if (base is not None and tensor.ndim == 2
                        and _is_excluded(key, model_type)):
                    excluded += 1
                continue

            # ── Convert to fp16 on device ──────────────────────────
            if tensor.dtype == torch.int8:
                s_key = f"{base}.weight_scale"
                if s_key in sd:
                    ws = sd[s_key].float().to(dev)
                    w = (tensor.float().to(dev)
                         * ws.view(-1, 1)).to(torch.float16)
                else:
                    w = tensor.float().to(dev).to(torch.float16)
            else:
                w = tensor.float().to(dev).to(torch.float16)

            # ── QuaRot rotation (offline) ──────────────────────────
            layer_quarot = False
            if H is not None and w.shape[1] % actual_gs == 0:
                try:
                    w = rotate_weight(w, H, group_size=actual_gs)
                    layer_quarot = True
                except ValueError:
                    pass

            out_f, in_f = w.shape

            # ── torchao INT4 quantize ──────────────────────────────
            tmp = nn.Linear(
                in_f, out_f, bias=False,
                device=dev, dtype=torch.float16)
            tmp.weight.data = w.clone()
            quantize_(tmp, Int4WeightOnlyConfig(
                group_size=actual_gs,
                int4_packing_format="plain_int32",
            ))

            # Check if quantize_ actually modified the weight
            if not hasattr(tmp.weight, 'qdata'):
                log.warning(
                    f"[TINT4] Layer {key} NOT quantized "
                    f"(in_f={in_f} incompatible with gs={actual_gs}), "
                    f"keeping original fp16")
                skipped += 1
                del w, tmp
                continue

            qt = tmp.weight

            # ── Write v7 format: weight key = int32 qdata ─────────
            sd[key] = qt.qdata.cpu()                            # int32
            sd[f"{base}.weight_scale"] = qt.scale.cpu()         # fp16
            sd[f"{base}.weight_zp"] = qt.zero_point.cpu()       # int8
            sd[f"{base}.weight_b0"] = torch.tensor(
                qt.block_size[0], dtype=torch.int32)
            sd[f"{base}.weight_b1"] = torch.tensor(
                qt.block_size[1], dtype=torch.int32)
            sd[f"{base}.comfy_quant"] = _make_comfy_quant_meta(
                quarot=layer_quarot,
                group_size=actual_gs if layer_quarot else None,
            )

            quantized += 1
            del w, tmp, qt

        if dev.type in ("xpu", "cuda"):
            try:
                (torch.xpu if dev.type == "xpu"
                 else torch.cuda).empty_cache()
            except Exception:
                pass

        # ── Strip text_encoders keys (FP8 models) ────────────────────
        stripped = 0
        for k in list(sd.keys()):
            if k.startswith("text_encoders."):
                del sd[k]
                stripped += 1
        if stripped:
            log.info(
                f"[TINT4] Stripped {stripped} text_encoders keys")

        # ── Global markers ──────────────────────────────────────────
        sd["__tint4_format__"] = torch.tensor(1, dtype=torch.uint8)
        sd["__tint4_quarot__"] = torch.tensor(
            1 if enable_quarot else 0, dtype=torch.uint8)
        sd["__tint4_group_size__"] = torch.tensor(
            actual_gs, dtype=torch.int32)
        sd["__tint4_model_type__"] = _str_tensor(model_type)

        save_kwargs = {"metadata": src_metadata} if src_metadata else {}
        log.info(f"[TINT4] Writing {dst_path} ...")
        comfy.utils.save_torch_file(sd, dst_path, **save_kwargs)

        log.info(
            f"\n{'='*60}\n"
            f"  TINT4 Quantization Complete (torchao INT4 v7)\n"
            f"  Model: {model_name} | Type: {model_type}\n"
            f"  Device: {dev} | QuaRot: {enable_quarot}"
            f" | gs={actual_gs}\n"
            f"  Quantized: {quantized} | Excluded: {excluded}"
            f" | Skipped: {skipped}\n"
            f"  Output: {dst_path}\n"
            f"  {'='*60}"
        )
        return ()


NODE_CLASS_MAPPINGS = {"TINT4ModelQuantizer": TINT4ModelQuantizer}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TINT4ModelQuantizer": "TINT4 Model Quantizer"}
