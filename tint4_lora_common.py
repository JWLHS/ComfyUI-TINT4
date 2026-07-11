"""
tint4_lora_common.py  v1.1
─────────────────────────────
Shared LoRA utilities — path normalization, format detection,
QuaRot, raw LoRA parsing, index-based reset, bypass signal I/O.

v1.1: +_read_clear_signal / _write_clear_signal (JS→Python bridge).
v1.0.2: reset handles pending/applied/legacy bake states.
"""
import logging
import torch
from .tint4_loader import TINT4Linear

log = logging.getLogger("TINT4-LoRA-Common")


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
	return path


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


def _rot_quarot_tensor(
	tensor: torch.Tensor,
	H: torch.Tensor | None,
	group_size: int,
	dev: torch.device,
) -> torch.Tensor:
	if H is None or group_size <= 0:
		return tensor
	if tensor.shape[1] % group_size != 0:
		return tensor
	Hd = H.to(tensor.device, dtype=torch.float16)
	ng = tensor.shape[1] // group_size
	return (tensor.reshape(tensor.shape[0], ng, group_size) @ Hd.T).reshape(
		tensor.shape[0], tensor.shape[1])


def _parse_raw_lora_sd(lora_sd: dict) -> dict[str, dict]:
	lora_data: dict[str, dict] = {}
	for key, tensor in lora_sd.items():
		if "lokr_w1" in key:
			idx = key.index("lokr_w1")
			lp = _normalize_layer_path(key[:idx].rstrip("."))
			if lp:
				lora_data.setdefault(lp, {})["lokr_w1"] = tensor
				lora_data[lp]["type"] = "lokr"
			continue
		if "lokr_w2" in key:
			idx = key.index("lokr_w2")
			lp = _normalize_layer_path(key[:idx].rstrip("."))
			if lp:
				lora_data.setdefault(lp, {})["lokr_w2"] = tensor
				lora_data[lp]["type"] = "lokr"
			continue
		if "lora_up" in key or "lora_B" in key:
			idx = (key.index("lora_up") if "lora_up" in key
				   else key.index("lora_B"))
			lp = _normalize_layer_path(key[:idx].rstrip("."))
			if lp:
				lora_data.setdefault(lp, {})["up"] = tensor
				if "type" not in lora_data[lp]:
					lora_data[lp]["type"] = "standard"
			continue
		if "lora_down" in key or "lora_A" in key:
			idx = (key.index("lora_down") if "lora_down" in key
				   else key.index("lora_A"))
			lp = _normalize_layer_path(key[:idx].rstrip("."))
			if lp:
				lora_data.setdefault(lp, {})["down"] = tensor
				if "type" not in lora_data[lp]:
					lora_data[lp]["type"] = "standard"
			continue
		if key.endswith(".alpha"):
			lp = _normalize_layer_path(key[:-6])
			if lp:
				t = tensor
				lora_data.setdefault(lp, {})["alpha"] = (
					float(t.mean()) if t.numel() > 1 else t.item())
			continue
	return lora_data


def _match_lora_to_index(
	lora_data: dict[str, dict],
	dm: torch.nn.Module,
	quarot_enabled: bool,
	group_size: int,
	H: torch.Tensor | None,
	dev: torch.device,
	cpu: torch.device = torch.device("cpu"),
) -> dict[str, list[dict]]:
	index = getattr(dm, '_tint4_lora_index', None)
	if index is None:
		log.warning("[TINT4] No index table")
		return {"quant": {}, "non_quant": {}}
	quant_entries: dict[str, list[dict]] = {}
	non_quant_entries: dict[str, list[dict]] = {}
	for lora_norm, info in lora_data.items():
		lora_type = info.get("type", "standard")
		targets: list[str] = []
		if lora_norm.endswith(".attn.qkv"):
			base = lora_norm.rsplit(".attn.qkv", 1)[0]
			targets = [f"{base}.attn.wq", f"{base}.attn.wk", f"{base}.attn.wv"]
		else:
			targets = [lora_norm]
		qkv_slices = [None]
		if len(targets) == 3:
			qkv_mod = index.get(lora_norm)
			if qkv_mod is not None:
				if isinstance(qkv_mod, TINT4Linear):
					out_f = qkv_mod.out_features
				elif hasattr(qkv_mod, 'weight') and qkv_mod.weight is not None:
					out_f = qkv_mod.weight.shape[0]
				else:
					out_f = 0
				if out_f > 0 and out_f % 3 == 0:
					hs = out_f // 3
					qkv_slices = [(0, hs), (hs, 2 * hs), (2 * hs, 3 * hs)]
		for ti, target in enumerate(targets):
			module = index.get(target)
			if module is None:
				continue
			is_quant = isinstance(module, TINT4Linear)
			if lora_type == "lokr":
				w1 = info.get("lokr_w1")
				w2 = info.get("lokr_w2")
				if w1 is None or w2 is None:
					continue
				w1_c = w1.to(cpu, torch.float16).clone()
				w2_c = w2.to(cpu, torch.float16).clone()
				w2_c = _rot_quarot_tensor(w2_c, H, group_size, dev)
				alpha_val = info.get("alpha", w1_c.shape[0])
				mult_base = alpha_val / max(w1_c.shape[0], 1)
				entry = {
					"type": "lokr", "lokr_w1": w1_c, "lokr_w2": w2_c,
					"alpha": alpha_val, "mult_base": mult_base,
					"factor": w1_c.shape[0],
					"slice": qkv_slices[ti] if ti < len(qkv_slices) else None,
				}
			else:
				down = info.get("down")
				up = info.get("up")
				if down is None or up is None:
					continue
				A = down.to(cpu, torch.float16).clone()
				B = up.to(cpu, torch.float16).clone()
				A = _rot_quarot_tensor(A, H, group_size, dev)
				alpha_val = info.get("alpha", up.shape[1])
				mult_base = alpha_val / max(up.shape[1], 1)
				entry = {
					"type": "standard", "down": A, "up": B,
					"alpha": alpha_val, "mult_base": mult_base,
					"slice": qkv_slices[ti] if ti < len(qkv_slices) else None,
				}
			if is_quant:
				quant_entries.setdefault(target, []).append(entry)
			else:
				non_quant_entries.setdefault(target, []).append(entry)
	return {"quant": quant_entries, "non_quant": non_quant_entries}


def _build_lora_entries_full(
	lora_sd: dict,
	dm: torch.nn.Module,
	quarot_enabled: bool,
	group_size: int,
	H: torch.Tensor | None,
	dev: torch.device,
) -> tuple[dict, str]:
	fmt = _auto_detect_format(lora_sd)
	if fmt == "bfl":
		lora_sd = _convert_bfl_to_standard(lora_sd)
	else:
		fmt = "standard"
	lora_data = _parse_raw_lora_sd(lora_sd)
	matched = _match_lora_to_index(
		lora_data, dm, quarot_enabled, group_size, H, dev)
	return matched, fmt


def _tint4_reset_all_loras(model) -> None:
	dm = model.model.diffusion_model
	while hasattr(dm, '_orig_mod'):
		dm = dm._orig_mod

	index = getattr(dm, '_tint4_lora_index', None)
	if index is not None:
		seen = set()
		for module in index.values():
			mid = id(module)
			if mid in seen:
				continue
			seen.add(mid)

			if hasattr(module, '_tint4_lora_entries'):
				object.__setattr__(module, '_tint4_lora_entries', None)

			bs = getattr(module, '_tint4_bake_state', None)
			if bs is None:
				continue

			hh = bs.pop('_hook_handle', None)
			if hh is not None:
				try:
					hh.remove()
				except Exception:
					pass
			bs.pop('_pending', None)

			applied = bs.pop('_applied', None)
			if applied is not None and hasattr(module, 'weight') and module.weight is not None:
				for delta_cpu, sl, se in applied:
					try:
						neg = (-delta_cpu).to(
							device=module.weight.device,
							dtype=module.weight.dtype)
						if sl is not None and se is not None:
							module.weight.data[sl:se].add_(neg)
						else:
							module.weight.data.add_(neg)
					except Exception:
						pass

			for key in list(bs.keys()):
				info = bs.pop(key, None)
				if isinstance(info, dict) and 'delta' in info:
					delta = info['delta']
					sl = info.get('sl')
					se = info.get('se')
					if hasattr(module, 'weight') and module.weight is not None:
						try:
							neg = (-delta).to(
								device=module.weight.device,
								dtype=module.weight.dtype)
							if sl is not None and se is not None:
								module.weight.data[sl:se].add_(neg)
							else:
								module.weight.data.add_(neg)
						except Exception:
							pass

			object.__setattr__(module, '_tint4_bake_state', None)
	else:
		log.warning("[TINT4] Reset called with no index table — skipped")

	object.__setattr__(model.model, '_tint4_loras', [])
	object.__setattr__(model.model, '_lora_needs_reset', False)


# ═══════════════════════════════════════════════════════════════════
# v1.1: bypass signal file I/O (JS → Python bridge)
# ═══════════════════════════════════════════════════════════════════

import os as _os
import json as _json

_SIGNAL_DIR = _os.path.join(_os.path.dirname(__file__), "lora_cache")
_SIGNAL_FILE = _os.path.join(_SIGNAL_DIR, "_signal.json")


def _read_clear_signal() -> bool:
	"""Read bypass signal from JS → return True if clear is needed."""
	try:
		if _os.path.exists(_SIGNAL_FILE):
			with open(_SIGNAL_FILE, "r", encoding="utf-8") as f:
				data = _json.load(f)
			_os.remove(_SIGNAL_FILE)
			if data.get("action") == "clear":
				log.info("[TINT4 Signal] Clear signal received — will force LoRA reset")
				return True
	except Exception:
		pass
	return False


def _write_clear_signal(payload: dict) -> None:
	"""Write signal file (called by HTTP endpoint /custom/TINT4/signal)."""
	try:
		_os.makedirs(_SIGNAL_DIR, exist_ok=True)
		with open(_SIGNAL_FILE, "w", encoding="utf-8") as f:
			_json.dump(payload, f)
	except Exception:
		pass
