"""
tint4_quantizer.py — TINT4 Model Quantizer  v8.1

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

v8.1: +16 native model types (SD3, Flux, PixArt, AuraFlow, Hydit,
	  HunyuanVideo, Mochi, Cosmos, CogVideoX, Lumina2, OmniGen2,
	  Lens, Kandinsky5, HiDream-O1, SeedVR2, Hunyuan3D).
	  +model_type_key() for user-friendly dropdown display names.
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


# ── 辅助：显示名 → 内部 key ─────────────────────────────────────────
def model_type_key(display_name: str) -> str:
	"""从下拉显示名提取内部 key。

	"flux (Flux.1 dev/schnell)" → "flux"
	"pixart (PixArt Alpha / Sigma)" → "pixart"
	"auto (Auto-detect)" → "auto"
	"""
	return display_name.split(" (")[0] if " (" in display_name else display_name


# ── Exclusion lists ──────────────────────────────────────────────────
# Keys containing any of these substrings are kept as fp16.
# Derived from model_detection.py heuristics + empirical testing.
_EXCLUSIONS = {
	# ── 已有 ───────────────────────────────────────────────────────
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
	"anima": [
		"adaln",
		"x_embedder",
		"final_layer",
		"t_embedder",
		"llm_adapter",
		"cross_attn",      # ← 加这个
	],

	# ── 新增：MMDiT 架构 ──────────────────────────────────────────
	"sd3": [
		# SD3 / SD3.5: joint_blocks x_block + context_block MMDiT
		# 排除输入投影、输出层、位置编码；量化 block 内 QKV + MLP
		"x_embedder", "y_embedder", "context_embedder",
		"final_layer", "pos_embed",
	],
	"flux": [
		# Flux.1 (非flux2): double_blocks + single_blocks MMDiT
		# img_mod.lin / txt_mod.lin / modulation.lin 是 adaLN 调制参数
		"img_in", "txt_in", "time_in", "vector_in",
		"guidance_in", "final_layer",
		"img_mod.lin", "txt_mod.lin", "modulation.lin",
	],
	"hunyuan_video": [
		# HunyuanVideo: MMDiT + 视频专用输入 (vision_in, byt5_in)
		"img_in", "txt_in", "time_in", "final_layer",
		"vector_in", "guidance_in", "vision_in", "byt5_in",
	],
	"hunyuan3d": [
		# Hunyuan3D 2.x: MMDiT, latent/cond 替代 img/txt
		"latent_in", "cond_in", "final_layer", "guidance_in",
	],

	# ── 新增：DiT 架构 ─────────────────────────────────────────────
	"auraflow": [
		# AuraFlow: double_layers + single_layers, cond_seq_linear 是条件投影
		"positional_encoding", "cond_seq_linear",
	],
	"hydit": [
		# Hunyuan DiT: 单流 DiT, mlp_t5 是 T5 文本投影
		"x_embedder", "extra_embedder", "final_layer",
		"time_embed", "mlp_t5",
	],
	"mochi": [
		# Mochi Preview: Genmo 视频 DiT
		"t5_yproj", "t5_yembed", "x_embed", "final_layer",
	],
	"pixart": [
		# PixArt Alpha / Sigma: 单流 DiT, t_block 是 time embedding
		"t_block", "pos_embed", "y_embedder",
		"ar_embedder", "x_embedder", "final_layer",
	],
	"cosmos": [
		# Cosmos: NVIDIA 视频 DiT, adaLN modulation
		"x_embedder", "final_layer", "adaln", "t_embedder",
	],
	"cogvideox": [
		# CogVideoX: patch_embed 含 Conv+Linear, ofs_embedding 是噪声增强
		"patch_embed", "proj_out", "ofs_embedding",
	],
	"lumina2": [
		# Lumina 2 / NewBie: cap_embedder 是 caption 嵌入
		"cap_embedder", "noise_refiner", "x_embedder",
		"final_layer", "t_embedder",
	],
	"omnigen2": [
		# OmniGen 2: time_caption_embed 是时间+文本条件嵌入
		"time_caption_embed", "x_embedder", "final_layer",
	],
	"lens": [
		# Lens: 图像 DiT, txt_norm 是文本特征归一化层
		"img_in", "proj_out", "txt_norm",
	],
	"kandinsky5": [
		# Kandinsky 5: visual + text 双 transformer
		"visual_embeddings", "time_embeddings",
	],
	"hidream_o1": [
		# HiDream-O1: t_embedder1 + x_embedder.proj1 是第一阶段嵌入
		"t_embedder1", "x_embedder.proj1", "final_layer",
	],
	"seedvr2": [
		# SeedVR 2: x_embedder 是 patch embedding
		"x_embedder", "final_layer",
	],

	"auto": [],
}

# 用户可见下拉列表 — model_type_key() 提取括号前内部 key
MODEL_TYPES = [
	# ── MMDiT ──
	"flux2 (Flux.2)",
	"flux (Flux.1 dev/schnell)",
	"sd3 (SD3 / SD3.5)",
	"hunyuan_video (Hunyuan Video)",
	"hunyuan3d (Hunyuan3D 2.x)",
	# ── DiT ──
	"pixart (PixArt Alpha / Sigma)",
	"hydit (Hunyuan DiT)",
	"auraflow (AuraFlow)",
	"mochi (Mochi Preview)",
	"cosmos (Cosmos)",
	"cogvideox (CogVideoX)",
	"lumina2 (Lumina 2 / NewBie)",
	"omnigen2 (OmniGen 2)",
	"lens (Lens)",
	"kandinsky5 (Kandinsky 5)",
	"hidream_o1 (HiDream-O1)",
	"seedvr2 (SeedVR 2)",
	# ── 已有 ──
	"z-image (Z-Image)",
	"chroma (Chroma / Radiance)",
	"wan (Wan 2.1)",
	"ltx2 (LTX Video 2)",
	"qwen (Qwen Image)",
	"ernie (Ernie Image)",
	"hidream (HiDream Full)",
	"boogu (Boogu)",
	"krea2 (Krea 2)",
	"ideogram4 (Ideogram 4)",
	"anima (Anima / Cosmos Predict2)",
	# ── 兜底 ──
	"auto (Auto-detect)",
]


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
				"model_type": (MODEL_TYPES, {
					"default": "flux2 (Flux.2)",
					"tooltip": "Must match the model architecture",
				}),
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
		"Quantize diffusion model to torchao INT4 (v8.1: +16 native DiT types)")

	def quantize(self, model_name, model_type, enable_quarot, group_size,
				 device, output_filename):
		# ── 显示名 → 内部 key ─────────────────────────────────────
		model_type = model_type_key(model_type)

		src_path = folder_paths.get_full_path(
			"diffusion_models", model_name)
		if src_path is None:
			raise FileNotFoundError(f"'{model_name}' not found")

		output_dir = folder_paths.get_output_directory()
		output_filename = os.path.basename(output_filename)  # ← 防止路径遍历
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

			# FP8 → FP16, applying per-tensor scale if present
			if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
				scale_key = f"{key.rsplit('.weight', 1)[0]}.weight_scale"
				scale = sd.get(scale_key)
				if scale is not None:
					sd[key] = (tensor.float() * scale.float()).to(torch.float16)
				else:
					sd[key] = tensor.to(torch.float16)
				tensor = sd[key]

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
