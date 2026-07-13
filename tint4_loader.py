"""
tint4_loader.py — TINT4 Model Loader v1.0 + v1.1 bypass fix

v1.1: load_model reads JS bypass signal + forces _tint4_reset_all_loras.
v8.4.0: +_build_tint4_lora_index (O(1) LoRA layer lookup),
		+_tint4_quarot_enabled / _tint4_group_size global marks,
		+_get_model_fingerprint for cache validation,
		+TINT4Linear.forward LoRA diagnostic log.
v8.3.1: _detach_cleanup now flushes cached _qt on all TINT4Linear
		layers so device VRAM is freed on model unload/swap.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc, os, json, hashlib
import logging
import folder_paths
import comfy.sd
import comfy.model_detection
import comfy.utils
import comfy.ops
from torchao.quantization.quantize_.workflows.int4.int4_plain_int32_tensor import (
	Int4PlainInt32Tensor,
)
from safetensors import safe_open

log = logging.getLogger("TINT4-Loader")

_orig_detect = comfy.model_detection.detect_unet_config

_QUANT_META_SUFFIXES = (
	".weight_scale", ".weight_zp", ".weight_b0", ".weight_b1",
	".weight_sh0", ".weight_sh1", ".comfy_quant",
)

_TUPLE_KEYS = {"patch_size", "window_size", "axes_dims", "axes_lens"}


def _normalize_index_path(name: str) -> str | None:
	for pf in ["diffusion_model.", "model.diffusion_model.", "model."]:
		if name.startswith(pf):
			name = name[len(pf):]
			break
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
	name = name.replace(".to_q", ".wq").replace(".to_k", ".wk")
	name = name.replace(".to_v", ".wv").replace(".to_out.0", ".wo")
	name = name.replace(".to_out", ".wo").replace(".to_gate", ".gate")
	name = name.replace(".q_proj", ".wq").replace(".k_proj", ".wk")
	name = name.replace(".v_proj", ".wv").replace(".out_proj", ".wo")
	name = name.replace(".self_attn.q", ".attn.wq")
	name = name.replace(".self_attn.k", ".attn.wk")
	name = name.replace(".self_attn.v", ".attn.wv")
	name = name.replace(".self_attn.o", ".attn.wo")
	name = name.replace(".attn.out", ".attn.wo")
	return name


def _build_tint4_lora_index(dm: nn.Module) -> dict:
	index: dict[str, nn.Module] = {}
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
			out_f = module.weight.shape[0] if hasattr(module, 'weight') else 0
			if out_f > 0 and out_f % 3 == 0:
				hs = out_f // 3
				base = norm.rsplit(".attn.qkv", 1)[0]
				index[f"{base}.attn.wq"] = module
				index[f"{base}.attn.wk"] = module
				index[f"{base}.attn.wv"] = module
		index[norm] = module
	return index


def _get_model_fingerprint(dm: nn.Module, quarot_enabled: bool,
						   group_size: int) -> str:
	rows = []
	for name, module in dm.named_modules():
		if not isinstance(module, TINT4Linear):
			continue
		rows.append(f"{name}:{module.in_features}:{module.out_features}")
	rows.sort()
	rows.append(f"quarot:{int(quarot_enabled)}")
	rows.append(f"gs:{group_size}")
	return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]


class TINT4Linear(nn.Module):

	def __init__(self, in_features, out_features, qdata, scale, zp,
				 block_size, bias=None):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		if bias is not None:
			self.bias = nn.Parameter(bias)
		else:
			self.register_parameter('bias', None)
		self._qdata = qdata
		self._scale = scale
		self._zp = zp
		self._block_size = block_size
		self._qt = None
		self._use_quarot: bool = False
		self._group_size: int = 128
		self._hadamard_H = None
		self._tint4_lora_entries: dict | None = None
		self._tint4_bake_state: dict | None = None

	def __del__(self):
		self._qdata = None
		self._scale = None
		self._zp = None
		self._qt = None
		self._tint4_lora_entries = None
		self._tint4_bake_state = None

	@property
	def weight(self):
		if self._qt is None:
			self._qt = Int4PlainInt32Tensor(
				self._qdata, self._scale, self._zp,
				self._block_size, [self.out_features, self.in_features],
			)
		return self._qt

	@weight.setter
	def weight(self, value):
		if isinstance(value, Int4PlainInt32Tensor):
			self._qt = value

	def release_xpu(self):
		self._qt = None

	def forward(self, x):
		x2 = x.reshape(-1, x.shape[-1])
		if self._use_quarot and self._hadamard_H is not None:
			try:
				from .wint8_quarot import rotate_activation
				x2 = rotate_activation(x2, self._hadamard_H, self._group_size)
			except Exception:
				pass
		dev = x.device
		if self._qt is None or self._qt.device != dev:
			self._qt = Int4PlainInt32Tensor(
				self._qdata.to(dev), self._scale.to(dev), self._zp.to(dev),
				self._block_size, [self.out_features, self.in_features],
			)
		out = F.linear(x2, self._qt, None)

		entries = self._tint4_lora_entries
		if entries is not None and len(entries) > 0:
			if not getattr(self, '_tint4_diag_printed', False):
				log.debug(
					f"[TINT4 Diag] FWD LoRA active: "
					f"keys={list(entries.keys())} "
					f"n_entries={sum(len(v) for v in entries.values())} "
					f"out_f={self.out_features}"
				)
				object.__setattr__(self, '_tint4_diag_printed', True)

			cd = (x.dtype if x.dtype in (torch.float16, torch.bfloat16)
				  else torch.float16)
			for lora_entries in entries.values():
				for e in lora_entries:
					if isinstance(e[0], str) and e[0] == "lokr":
						_, w1, w2, mult, factor = e[:5]
						sl = e[5] if len(e) > 5 else None
						se = e[6] if len(e) > 6 else None
						of2, if2 = w2.shape
						w1d = w1.to(device=dev, dtype=cd)
						w2d = w2.to(device=dev, dtype=cd)
						w1x = w1d.repeat_interleave(
							of2 // factor, dim=0).repeat_interleave(
							if2 // factor, dim=1)
						dw = (w1x * w2d).mul_(mult)
						lo = x2.to(cd) @ dw.T
						if sl is not None:
							if lo.shape[1] != (se - sl):
								continue
							out[:, sl:se] += lo
						else:
							if lo.shape != out.shape:
								continue
							out += lo
						continue
					A, B, mult = e[:3]
					sl = e[3] if len(e) > 3 else None
					se = e[4] if len(e) > 4 else None
					Ad = A.to(device=dev, dtype=cd)
					Bd = B.to(device=dev, dtype=cd)
					lo = (x2.to(cd) @ Ad.T) @ Bd.T * mult
					if sl is not None:
						if lo.shape[1] != (se - sl):
							continue
						out[:, sl:se] += lo
					else:
						if lo.shape[1] != out.shape[1]:
							continue
						out += lo
		if self.bias is not None:
			out += self.bias.to(device=dev, dtype=out.dtype)
		return out.reshape(*x.shape[:-1], out.shape[-1])


def _detect_krea2(sd: dict):
	keys = list(sd.keys())
	if "first.weight" not in sd:
		return None
	pe = sd["first.weight"]
	dim, in_ch = pe.shape[0], pe.shape[1]
	mlp_key = "blocks.0.mlp.up.weight"
	ffn_dim = dim * 4
	if mlp_key in sd:
		w = sd[mlp_key]
		ffn_dim = w.shape[0] if w.dtype != torch.int32 else w.shape[1] * 8
	num_layers = comfy.model_detection.count_blocks(keys, "blocks." + "{}.")
	cfg = {
		"image_model": "krea2", "dim": dim, "in_dim": in_ch,
		"patch_size": 2, "out_dim": in_ch, "num_heads": dim // 128,
		"num_layers": num_layers, "ffn_dim": ffn_dim,
	}
	if "txt_in.weight" in keys:
		cfg["txt_in"] = True
	return cfg


def _detect_boogu(sd: dict):
	keys = list(sd.keys())
	x_key = "x_embedder.weight"
	if x_key not in sd:
		return None
	hidden_size = sd[x_key].shape[0]
	num_layers = comfy.model_detection.count_blocks(
		keys, "single_stream_layers." + "{}.")
	num_double = comfy.model_detection.count_blocks(
		keys, "double_stream_layers." + "{}.")
	num_refiner = comfy.model_detection.count_blocks(
		keys, "noise_refiner." + "{}.")
	cap_key = "time_caption_embed.caption_embedder.0.weight"
	instr_dim = (sd[cap_key].shape[0]
				 if cap_key in sd else hidden_size)
	return {
		"image_model": "boogu",
		"hidden_size": hidden_size,
		"num_layers": num_layers,
		"num_double_stream_layers": num_double,
		"num_refiner_layers": num_refiner,
		"instruction_feat_dim": instr_dim,
	}


def _detect_wan(sd: dict, key_prefix: str = ""):
	keys = list(sd.keys())
	pe_key = f"{key_prefix}patch_embedding.weight"
	if pe_key not in keys:
		return None
	pe = sd[pe_key]
	if pe.ndim != 5 or pe.shape[1] != 16:
		return None
	ffn_key = f"{key_prefix}blocks.0.ffn.0.weight"
	if ffn_key not in keys:
		return None
	dim = int(pe.shape[0])
	ffn_dim = int(sd[ffn_key].shape[0])
	num_layers = comfy.model_detection.count_blocks(
		keys, f"{key_prefix}blocks." + "{}.")
	cfg = {
		"image_model": "wan2.1", "dim": dim, "out_dim": 16,
		"num_heads": dim // 128, "ffn_dim": ffn_dim,
		"num_layers": num_layers, "patch_size": (1, 2, 2),
		"freq_dim": 256, "window_size": (-1, -1),
		"qk_norm": True, "cross_attn_norm": True, "eps": 1e-6,
		"in_dim": int(pe.shape[1]),
	}
	cfg["model_type"] = (
		"i2v" if f"{key_prefix}img_emb.proj.0.bias" in keys else "t2v")
	return cfg


def _detect_fallback(sd, key_prefix, metadata=None, *, model_type=None):
	if model_type == "krea2":
		cfg = _detect_krea2(sd)
		if cfg is not None:
			return cfg
	if model_type == "boogu":
		cfg = _detect_boogu(sd)
		if cfg is not None:
			return cfg
	if model_type == "wan":
		cfg = _detect_wan(sd, key_prefix)
		if cfg is not None:
			return cfg
	return _orig_detect(sd, key_prefix, metadata)


class TINT4ModelLoader:
	NAME = "TINT4 Model Loader"
	CATEGORY = "TINT4"

	@classmethod
	def INPUT_TYPES(cls):
		from .tint4_quantizer import MODEL_TYPES
		return {
			"required": {
				"unet_name": (
					folder_paths.get_filename_list("diffusion_models"),
					{"tooltip": "TINT4 model from TINT4ModelQuantizer"},
				),
				"model_type": (
					MODEL_TYPES,
					{"default": "flux2",
					 "tooltip": "Must match quantization type"},
				),
			},
		}

	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_model"

	def load_model(self, unet_name, model_type):
		# v1.1: read JS bypass signal (dual channel: HTTP + graphToPrompt)
		from .tint4_lora_common import _read_clear_signal, _tint4_reset_all_loras
		force_reset = _read_clear_signal()
		if force_reset:
			log.info("[TINT4] ⚠️ Bypass signal received — will force LoRA reset after load")

		unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
		if unet_path is None:
			raise FileNotFoundError(f"[TINT4] '{unet_name}' not found")

		log.info(f"[TINT4] Loading: {unet_name} (type={model_type})")

		sd: dict = {}
		quant_map: dict = {}
		_quant_specs: list = []
		is_tint4 = False
		is_quarot = False
		quarot_gs = 128

		with safe_open(unet_path, framework="pt") as f:
			for k in f.keys():
				if k == "__tint4_format__":
					is_tint4 = True; continue
				if k == "__tint4_quarot__":
					is_quarot = (f.get_tensor(k).item() == 1); continue
				if k == "__tint4_group_size__":
					quarot_gs = f.get_tensor(k).item(); continue
				if k == "__tint4_model_type__":
					continue

				if k.endswith(".weight_qdata"):
					orig_base = k.rsplit(".weight_qdata", 1)[0]
					try:
						sh0 = f.get_tensor(f"{orig_base}.weight_sh0").item()
						sh1 = f.get_tensor(f"{orig_base}.weight_sh1").item()
					except (KeyError, ValueError):
						continue
					_quant_specs.append((orig_base, sh0, sh1))
					quant_map[orig_base] = {
						"qdata": f.get_tensor(k),
						"scale": f.get_tensor(f"{orig_base}.weight_scale"),
						"zp":    f.get_tensor(f"{orig_base}.weight_zp"),
						"b0":    f.get_tensor(f"{orig_base}.weight_b0").item(),
						"b1":    f.get_tensor(f"{orig_base}.weight_b1").item(),
						"sh0":   sh0, "sh1": sh1,
					}
					continue
				if k.endswith(_QUANT_META_SUFFIXES):
					continue
				v = f.get_tensor(k)
				if k.endswith(".weight") and v.dtype == torch.int32:
					orig_base = k.rsplit(".weight", 1)[0]
					try:
						s = f.get_tensor(f"{orig_base}.weight_scale")
						z = f.get_tensor(f"{orig_base}.weight_zp")
						b0 = f.get_tensor(f"{orig_base}.weight_b0")
						b1 = f.get_tensor(f"{orig_base}.weight_b1")
					except Exception:
						sd[k] = v; continue
					sh0, sh1 = v.shape[0], v.shape[1] * 8
					_quant_specs.append((orig_base, sh0, sh1))
					quant_map[orig_base] = {
						"qdata": v, "scale": s, "zp": z,
						"b0": b0.item(), "b1": b1.item(),
						"sh0": sh0, "sh1": sh1,
					}
					continue
				sd[k] = v

		if not is_tint4:
			raise ValueError("[TINT4] Not a TINT4 model (missing __tint4_format__)")

		cache_path = unet_path + ".tint4_config.json"
		cached_config = None

		if os.path.exists(cache_path):
			try:
				with open(cache_path, "r") as f:
					cached = json.load(f)
				if (cached.get("model_type") == model_type
						and cached.get("quant_layers") == len(quant_map)):
					cached_config = cached["unet_config"]
					for key in _TUPLE_KEYS:
						if key in cached_config and isinstance(
								cached_config[key], list):
							cached_config[key] = tuple(cached_config[key])
					log.info("[TINT4] Using cached unet_config")
				else:
					log.info("[TINT4] Cache mismatch, re-detecting")
			except Exception as e:
				log.warning(f"[TINT4] Failed to read cache: {e}")

			from .tint4_aimdo import build_weight_placeholders
			n_placeholders, placeholder_note = build_weight_placeholders(
				_quant_specs, sd)

			log.info(
				f"[TINT4] QuaRot={'ON' if is_quarot else 'OFF'}"
				f"  gs={quarot_gs}  "
				f"{len(quant_map)} quant layers, {len(sd)} sd keys"
				f"  ({n_placeholders} weight placeholders)  {placeholder_note}"
			)
			del _quant_specs; gc.collect()

		H = None
		if is_quarot:
			from .wint8_quarot import build_hadamard
			H = build_hadamard(quarot_gs, device="cpu", dtype=torch.float32)

		_captured = {}

		def _detect_wrapper(sd_in, key_prefix, metadata=None):
			if cached_config is not None:
				return dict(cached_config)
			result = _detect_fallback(
				sd_in, key_prefix, metadata=metadata,
				model_type=model_type)
			_captured["config"] = result
			return result

		comfy.model_detection.detect_unet_config = _detect_wrapper
		try:
			model = comfy.sd.load_diffusion_model_state_dict(
				sd, model_options={
					"custom_operations": comfy.ops.manual_cast,
				}, metadata={})
		finally:
			comfy.model_detection.detect_unet_config = _orig_detect

		del sd; gc.collect()

		if cached_config is None and _captured.get("config"):
			try:
				config_to_save = dict(_captured["config"])
				with open(cache_path, "w") as f:
					json.dump({
						"unet_config": config_to_save,
						"quant_layers": len(quant_map),
						"model_type": model_type,
					}, f, indent=2, default=list)
				log.info(f"[TINT4] Cached config → {cache_path}")
			except Exception as e:
				log.warning(f"[TINT4] Failed to save cache: {e}")

		dm = model.model.diffusion_model
		while hasattr(dm, "_orig_mod"):
			dm = dm._orig_mod

		replacements = []
		for parent_name, parent_mod in dm.named_modules():
			for child_name, child_mod in parent_mod.named_children():
				if not isinstance(child_mod, nn.Linear):
					continue
				full = (f"{parent_name}.{child_name}"
						if parent_name else child_name)
				for c in [f"diffusion_model.{full}",
						  f"model.diffusion_model.{full}",
						  f"model.{full}", full]:
					if c in quant_map:
						replacements.append((parent_mod, child_name, c))
						break

		saved_biases, freed = {}, 0
		for pm, cn, bk in replacements:
			old = getattr(pm, cn)
			saved_biases[bk] = (
				old.bias.data.clone()
				if old.bias is not None and old.bias.numel() > 0
				else None)
			if hasattr(old, 'weight') and old.weight is not None:
				freed += old.weight.numel() * old.weight.element_size()
				old.weight = nn.Parameter(torch.empty(0, device='cpu'))
			if old.bias is not None and old.bias.numel() > 0:
				freed += old.bias.numel() * old.bias.element_size()
				old.bias = nn.Parameter(torch.empty(0, device='cpu'))
		gc.collect()
		log.info(f"[TINT4] Released {freed / 1024**3:.2f} GB fp16 weights")

		injected = 0
		for pm, cn, bk in replacements:
			q = quant_map.pop(bk)
			bias = saved_biases.get(bk)
			nm = TINT4Linear(
				q["sh1"], q["sh0"],
				q["qdata"], q["scale"], q["zp"],
				[q["b0"], q["b1"]], bias=bias,
			)
			if is_quarot and nm.in_features % quarot_gs == 0:
				nm._use_quarot = True
				nm._group_size = quarot_gs
				nm._hadamard_H = H
			setattr(pm, cn, nm)
			injected += 1

		del quant_map, saved_biases; gc.collect()

		dm._tint4_lora_index = _build_tint4_lora_index(dm)
		dm._tint4_quarot_enabled = is_quarot
		dm._tint4_group_size = quarot_gs
		dm._tint4_fingerprint = _get_model_fingerprint(dm, is_quarot, quarot_gs)
		log.info(
			f"[TINT4] LoRA index: {len(dm._tint4_lora_index)} entries "
			f"(QuaRot={'ON' if is_quarot else 'OFF'}, "
			f"fingerprint={dm._tint4_fingerprint})"
		)

		try:
			mp = model.model
			if hasattr(mp, 'weights'):
				mp.weights = list(mp.model.parameters())
			bm = getattr(mp, 'model', None)
			if bm is not None:
				for a in ('dynamic_vbars', 'dynamic_pins'):
					d = getattr(bm, a, None)
					if isinstance(d, dict):
						d.clear()
			gc.collect()
		except Exception:
			pass

		log.info(f"[TINT4] Injected {injected} TINT4Linear layers")

		from .tint4_aimdo import patch_model_for_aimdo
		patch_model_for_aimdo(model)

		# v1.1: guaranteed LoRA cleanup after model load
		# Handles: bypass residue, middle-node interference, stale model refs
		_tint4_reset_all_loras(model)
		if force_reset:
			log.info("[TINT4] ✓ LoRA state force-cleared after model load")

		log.info(
			f"[TINT4] Loaded '{unet_name}' | {injected} INT4 layers"
		)
		return (model,)


NODE_CLASS_MAPPINGS = {"TINT4ModelLoader": TINT4ModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TINT4ModelLoader": "TINT4 Model Loader"}
