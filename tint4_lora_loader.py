"""
tint4_lora_loader.py — TINT4 LoRA Loader v1.0.3

Single LoRA injection with lightweight JSON cache + index-based O(1) matching.

v1.0.4: runtime B-slice fallback in bake pre-hook (QKV 11520→3840).
v1.0.3: fix QKV _inject_bake B slice.
v1.0.2: bake delta computed on GPU in pre-hook.
v1.0.1: fix mult_base default.
v1.0.0: initial release.
"""
import time
import logging
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import folder_paths
import comfy.utils
import comfy.patcher_extension
from .tint4_lora_common import (
	_tint4_reset_all_loras,
	_auto_detect_format,
	_convert_bfl_to_standard,
	_parse_raw_lora_sd,
	_get_accelerator_device,
	_rot_quarot_tensor,
)
from .tint4_loader import TINT4Linear
from .tint4_lora_cache import load_lora_cache, save_lora_cache

log = logging.getLogger("TINT4-LoRA")


# ── Pruned-base adaln grid injection (adapted from ComfyUI-MiniMax-H3-Turbo,
#    Apache-2.0): on pruned H3 the adaln update lives in the 2688-dim
#    silu(t_emb) space, so it can't be a weight patch.  We re-inject it at run
#    time via a shared E-grid + forward-attribute patches on each block's
#    adaln_proj (which the TINT4 model keeps per-block).
_SHIFT_V, _SHIFT_A = 12.0, 3.0
_EGRID_CACHE = None


def _egrid_tint4():
    global _EGRID_CACHE
    if _EGRID_CACHE is None:
        p = os.path.join(os.path.dirname(__file__), "..", "ComfyUI-MiniMax-H3-Turbo",
                         "h3_silu_temb_grid.safetensors")
        p = os.path.abspath(p)
        if not os.path.exists(p):
            raise FileNotFoundError(f"E-grid missing: {p}")
        _EGRID_CACHE = comfy.utils.load_torch_file(p)["silu_t_emb_grid"]  # [1025, 2688]
    return _EGRID_CACHE


def _time_shift_sigma_tint4(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _unique_t_tint4(timestep, shift_v, shift_a, has_vis_cond):
    sv = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
    t_v = 1.0 - sv
    t_a = 1.0 - _time_shift_sigma_tint4(sv, shift_v, shift_a)
    s = {t_v, t_a}
    if has_vis_cond:
        s.add(max(t_v, 0.999))
    return sorted(s)


def _interp_egrid_tint4(unique_t, E, device, dtype):
    E = E.to(device)
    n = E.shape[0]
    rows = []
    for t in unique_t:
        pos = min(max(t, 0.0), 1.0) * (n - 1)
        i0 = min(int(math.floor(pos)), n - 2)
        rows.append(torch.lerp(E[i0].float(), E[i0 + 1].float(), pos - i0))
    return torch.stack(rows).to(dtype)  # [M, 2688]


def _make_adaln_forward_tint4(base, a, b, shared):
    """Returns a replacement AdalnProj.forward that adds B @ A @ silu(t_emb)."""
    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        st = shared.get("silu_temb")
        if st is not None:
            av = a.to(x.device, x.dtype)
            bv = b.to(x.device, x.dtype)
            sv = st.to(x.device, x.dtype)
            x = x + (bv @ (av @ sv.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)
    return forward


def _inject_adaln_egrid_tint4(new_model, dm, lora, adaln, strength):
    E = _egrid_tint4()
    shared = {"silu_temb": None}
    shift_v = float(getattr(dm, "sigma_shift_video", _SHIFT_V))
    shift_a = float(getattr(dm, "sigma_shift_audio", _SHIFT_A))

    def wrap(executor, *args, **kwargs):
        ts = args[1] if len(args) > 1 else kwargs.get("timestep")
        ctx = args[2] if len(args) > 2 else kwargs.get("context")
        payload = kwargs.get("minimax_payload") or {}
        has_vc = bool(payload.get("keyframes") or payload.get("refs"))
        us = _unique_t_tint4(ts, shift_v, shift_a, has_vc)
        shared["silu_temb"] = _interp_egrid_tint4(us, E, ctx.device, ctx.dtype)
        return executor(*args, **kwargs)

    new_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3turbo", wrap)
    for name in adaln:
        a = lora[name + ".lora_A.weight"]
        b = lora[name + ".lora_B.weight"] * strength
        key = "diffusion_model." + name.rsplit(".linear", 1)[0]
        new_model.add_object_patch(
            key + ".forward",
            _make_adaln_forward_tint4(new_model.get_model_object(key), a, b, shared))
    return len(adaln)

_INJECT_TIMING_THRESHOLD = 3.0


def _infer_qkv_slice(target_path: str, module, index: dict) -> tuple | None:
	if not isinstance(module, TINT4Linear):
		return None
	out_f = module.out_features
	if out_f % 3 != 0:
		return None
	if target_path.endswith(".attn.wq"):
		head_idx = 0
		qkv_path = target_path[:-len(".attn.wq")] + ".attn.qkv"
	elif target_path.endswith(".attn.wk"):
		head_idx = 1
		qkv_path = target_path[:-len(".attn.wk")] + ".attn.qkv"
	elif target_path.endswith(".attn.wv"):
		head_idx = 2
		qkv_path = target_path[:-len(".attn.wv")] + ".attn.qkv"
	else:
		return None
	qkv_mod = index.get(qkv_path)
	if qkv_mod is not module:
		return None
	hs = out_f // 3
	return (head_idx * hs, (head_idx + 1) * hs)


def _resolve_qkv_slices(index: dict, norm: str) -> list[tuple[str, tuple | None]]:
	base = norm.rsplit(".attn.qkv", 1)[0]
	qkv_mod = index.get(norm)
	if qkv_mod is not None:
		if isinstance(qkv_mod, TINT4Linear):
			out_f = qkv_mod.out_features
		elif hasattr(qkv_mod, 'weight') and qkv_mod.weight is not None:
			out_f = qkv_mod.weight.shape[0]
		else:
			out_f = 0
		if out_f > 0 and out_f % 3 == 0:
			hs = out_f // 3
			return [
				(f"{base}.attn.wq", (0, hs)),
				(f"{base}.attn.wk", (hs, 2 * hs)),
				(f"{base}.attn.wv", (2 * hs, 3 * hs)),
			]
		return [
			(f"{base}.attn.wq", None),
			(f"{base}.attn.wk", None),
			(f"{base}.attn.wv", None),
		]
	for probe_key in [f"{base}.attn.wq", f"{base}.attn.wk", f"{base}.attn.wv"]:
		mod = index.get(probe_key)
		if mod is None:
			continue
		if isinstance(mod, TINT4Linear):
			out_f = mod.out_features
		elif hasattr(mod, 'weight') and mod.weight is not None:
			out_f = mod.weight.shape[0]
		else:
			continue
		if out_f > 0:
			return [
				(f"{base}.attn.wq", (0, out_f)),
				(f"{base}.attn.wk", (out_f, 2 * out_f)),
				(f"{base}.attn.wv", (2 * out_f, 3 * out_f)),
			]
		break
	return [
		(f"{base}.attn.wq", None),
		(f"{base}.attn.wk", None),
		(f"{base}.attn.wv", None),
	]


# ═══════════════════════════════════════════════════════════════
# v1.0.4: runtime B-slice fallback in pre-hook
# ═══════════════════════════════════════════════════════════════

def _make_bake_pre_hook(module: nn.Module):
	def _bake_pre_hook(_mod, _inputs):
		bs = getattr(module, '_tint4_bake_state', None)
		if bs is None:
			return
		pending = bs.get('_pending')
		if not pending:
			return
		w_dev = module.weight.device
		w_dtype = module.weight.dtype
		cpu = torch.device("cpu")
		applied = []
		try:
			for A_cpu, B_cpu, mult, sl, se in pending:
				if sl is not None and se is not None and B_cpu.shape[0] != (se - sl):
					B_cpu = B_cpu[sl:se].contiguous()
				A_gpu = A_cpu.to(device=w_dev, dtype=w_dtype)
				B_gpu = B_cpu.to(device=w_dev, dtype=w_dtype)
				# LoRA files disagree on the down/up convention: standard
				# diffusers saves A=[rank,in], B=[out,rank] (delta = B@A);
				# some (e.g. minimax_h3_turbo adaln) save A=[in,rank],
				# B=[out,rank] (delta = B@A.T).  Try the direct product first,
				# then the transposed fallback.
				try:
					delta_gpu = (B_gpu @ A_gpu).mul_(mult)
					if delta_gpu.shape != module.weight.shape:
						raise RuntimeError("shape mismatch, retry transposed")
				except Exception:
					delta_gpu = (B_gpu @ A_gpu.T).mul_(mult)
				if sl is not None and se is not None:
					if delta_gpu.shape[0] != (se - sl):
						delta_gpu = delta_gpu[sl:se].contiguous()
					module.weight.data[sl:se].add_(delta_gpu)
				else:
					if delta_gpu.shape[0] != module.weight.shape[0]:
						delta_gpu = delta_gpu[:module.weight.shape[0]].contiguous()
					module.weight.data.add_(delta_gpu)
				applied.append((delta_gpu.to(device=cpu, dtype=torch.float16).clone(), sl, se))
		except Exception as e:
			log.warning(
				f"[TINT4 LoRA] bake pre-hook failed: {e} | "
				f"mod={type(module).__name__} w={tuple(module.weight.shape)} "
				f"A={tuple(A_cpu.shape)} B={tuple(B_cpu.shape)}"
			)
		bs.pop('_pending', None)
		bs['_applied'] = applied
		hh = bs.pop('_hook_handle', None)
		if hh is not None:
			hh.remove()
	return _bake_pre_hook


class TINT4LoRALoader:
	NAME = "TINT4 LoRA Loader"
	CATEGORY = "TINT4"

	_prev_changed = None

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
	def IS_CHANGED(cls, model, lora_name, strength):
		import random
		current = (lora_name, strength, random.random())
		cls._prev_changed = current
		return current

	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_lora"

	def load_lora(self, model, lora_name, strength):
		if getattr(model.model, '_lora_needs_reset', False):
			_tint4_reset_all_loras(model)
			object.__setattr__(model.model, '_lora_needs_reset', False)
		if abs(strength) < 1e-5:
			return (model,)

		lora_path = folder_paths.get_full_path("loras", lora_name)
		if lora_path is None:
			raise FileNotFoundError(f"[TINT4 LoRA] '{lora_name}' not found")

		dm = model.model.diffusion_model
		while hasattr(dm, '_orig_mod'):
			dm = dm._orig_mod

		quarot_enabled = getattr(dm, '_tint4_quarot_enabled', False)
		group_size = getattr(dm, '_tint4_group_size', 128)
		index = getattr(dm, '_tint4_lora_index', None) or {}
		dev = _get_accelerator_device()
		cpu = torch.device("cpu")

		H = None
		if quarot_enabled:
			from .wint8_quarot import build_hadamard
			H = build_hadamard(group_size, device="cpu", dtype=torch.float32)

		t0 = time.perf_counter()
		cached = load_lora_cache(lora_path)
		lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

		fmt = _auto_detect_format(lora_sd)
		if fmt == "bfl":
			lora_sd = _convert_bfl_to_standard(lora_sd)
		else:
			fmt = "standard"

		cacheable_keys = list(lora_sd.keys())
		lora_data = _parse_raw_lora_sd(lora_sd)

		aq, ab = self._inject_from_lora_data(
			dm, index, lora_sd, lora_data, lora_name, strength,
			quarot_enabled, group_size, H, dev, cpu)

		adaln_injected = 0
		try:
			if getattr(dm, "use_adaln_curves", False):
				adaln_names = sorted({k.rsplit(".lora_A.weight", 1)[0] for k in lora_sd
				                      if k.endswith("adaln_proj.linear.lora_A.weight")})
				if adaln_names:
					adaln_injected = _inject_adaln_egrid_tint4(
						model, dm, lora_sd, adaln_names, strength)
		except Exception as _e:
			log.warning(f"[TINT4 LoRA] adaln grid injection failed: {_e}")

		if cached is None:
			save_lora_cache(lora_path, fmt, cacheable_keys)

		elapsed = time.perf_counter() - t0
		tag = "cached" if cached else "full load"
		log.info(
			f"[TINT4 LoRA] ✓ {lora_name} | {tag} | "
			f"{aq} quant + {ab} bake-in + {adaln_injected} adaln-grid | "
			f"strength={strength} | {elapsed:.2f}s"
		)

		del lora_sd, lora_data

		if not hasattr(model.model, '_tint4_loras'):
			object.__setattr__(model.model, '_tint4_loras', [])
		model.model._tint4_loras.append({
			"name": lora_name, "strength": strength, "path": lora_path,
		})
		return (model,)

	def _inject_from_lora_data(
		self, dm, index, lora_sd, lora_data, lora_name, strength,
		quarot_enabled, group_size, H, dev, cpu,
	):
		aq, ab = 0, 0
		layer_times = []
		inject_t0 = time.perf_counter()

		for norm, info in lora_data.items():
			t_layer = time.perf_counter()
			lora_type = info.get("type", "standard")

			# Pruned H3: adaln keys are re-injected at runtime via the E-grid
			# (see _inject_adaln_egrid_tint4); never weight-patch them here.
			if "adaln_proj" in norm:
				log.debug(f"[TINT4 LoRA] adaln skipped for weight-path: {norm}")
				continue

			if norm.endswith(".attn.qkv"):
				targets = _resolve_qkv_slices(index, norm)
			else:
				targets = [(norm, None)]

			popped = set()
			for target_path, qkv_slice in targets:
				module = index.get(target_path)
				if module is None:
					continue

				if qkv_slice is None:
					qkv_slice = _infer_qkv_slice(target_path, module, index)

				is_quant = isinstance(module, TINT4Linear)
				alpha_val = info.get("alpha")

				mid = id(module)
				if mid not in popped:
					self._pop_module_lora(module, lora_name)
					popped.add(mid)

				if lora_type == "lokr":
					w1 = info.get("lokr_w1")
					w2 = info.get("lokr_w2")
					if w1 is None or w2 is None:
						continue
					self._inject_lokr(
						module, lora_name, w1, w2,
						alpha_val, strength, qkv_slice,
						quarot_enabled, H, group_size, dev, cpu)
					aq += 1
				else:
					down = info.get("down")
					up = info.get("up")
					if down is None or up is None:
						continue
					if is_quant:
						self._inject_quant(
							module, lora_name, down, up,
							alpha_val, strength, qkv_slice,
							quarot_enabled, H, group_size, dev, cpu)
						aq += 1
					else:
						self._inject_bake(
							module, lora_name, down, up,
							alpha_val, strength, qkv_slice, cpu)
						ab += 1
			dt = time.perf_counter() - t_layer
			layer_times.append((norm, dt))

		total_inject = time.perf_counter() - inject_t0
		if total_inject > _INJECT_TIMING_THRESHOLD:
			layer_times.sort(key=lambda x: -x[1])
			avg = total_inject / max(len(layer_times), 1)
			log.warning(
				f"[TINT4 LoRA] ⏱ inject: total={total_inject:.2f}s, "
				f"avg={avg*1000:.1f}ms/layer, n={len(layer_times)}"
			)
			for name, dt in layer_times[:5]:
				if dt > 0.05:
					log.warning(
						f"[TINT4 LoRA]   slow: {name} → {dt*1000:.1f}ms"
					)
		return aq, ab

	@staticmethod
	def _pop_module_lora(module, lora_name):
		le = getattr(module, '_tint4_lora_entries', None)
		if le is not None:
			le.pop(lora_name, None)
		bs = getattr(module, '_tint4_bake_state', None)
		if bs is not None:
			applied = bs.pop('_applied', None)
			if applied is not None and hasattr(module, 'weight') and module.weight is not None:
				for delta_cpu, sl, se in applied:
					try:
						neg = (-delta_cpu).to(device=module.weight.device, dtype=module.weight.dtype)
						if sl is not None and se is not None:
							module.weight.data[sl:se].add_(neg)
						else:
							module.weight.data.add_(neg)
					except Exception:
						pass
			bs.pop(lora_name, None)
			bs.pop('_pending', None)
			hh = bs.pop('_hook_handle', None)
			if hh is not None:
				try:
					hh.remove()
				except Exception:
					pass

	def _inject_quant(
		self, module, lora_name, down, up, alpha_val,
		strength, qkv_slice, quarot_enabled, H, group_size,
		dev, cpu,
	):
		A = down.to(cpu, torch.float16).clone()
		B = up.to(cpu, torch.float16).clone()
		A = _rot_quarot_tensor(A, H, group_size, dev) if quarot_enabled else A
		rank = up.shape[1] if up.ndim >= 2 else 1
		mult_base = (alpha_val / max(rank, 1)) if alpha_val else 1.0
		mult = mult_base * strength
		le = getattr(module, '_tint4_lora_entries', None)
		if le is None:
			le = {}
			object.__setattr__(module, '_tint4_lora_entries', le)
		if qkv_slice is not None:
			sl, se = qkv_slice
			B_sliced = B[sl:se].contiguous().clone()
			le.setdefault(lora_name, []).append(
				(A, B_sliced, mult, 0, se - sl))
		else:
			le.setdefault(lora_name, []).append((A, B, mult))

	def _inject_bake(
		self, module, lora_name, down, up, alpha_val,
		strength, qkv_slice, cpu,
	):
		if not hasattr(module, 'weight') or module.weight is None:
			return
		A = down.to(cpu, torch.float16).clone()
		B = up.to(cpu, torch.float16).clone()
		if qkv_slice is not None:
			B = B[qkv_slice[0]:qkv_slice[1]].contiguous().clone()
		rank = up.shape[1] if up.ndim >= 2 else 1
		mult_base = (alpha_val / max(rank, 1)) if alpha_val else 1.0
		mult = mult_base * strength

		bs = getattr(module, '_tint4_bake_state', None)
		if bs is None:
			bs = {}
			object.__setattr__(module, '_tint4_bake_state', bs)
		pending = bs.get('_pending')
		if pending is None:
			pending = []
			bs['_pending'] = pending
		sl = qkv_slice[0] if qkv_slice else None
		se = qkv_slice[1] if qkv_slice else None
		pending.append((A, B, mult, sl, se))
		if '_hook_handle' not in bs:
			hook = module.register_forward_pre_hook(_make_bake_pre_hook(module))
			bs['_hook_handle'] = hook

	def _inject_lokr(
		self, module, lora_name, w1, w2, alpha_val,
		strength, qkv_slice, quarot_enabled, H, group_size,
		dev, cpu,
	):
		w1_c = w1.to(cpu, torch.float16).clone()
		w2_c = w2.to(cpu, torch.float16).clone()

		# ── QKV cache: avoid kronning same w1/w2 3 times ──
		cache = getattr(module, '_tint4_lokr_kron_cache', None)
		cache_key = (w1_c.shape, w2_c.shape)
		if cache is not None and cache.get('key') == cache_key:
			delta = cache['delta']
		else:
			delta = torch.kron(w1_c, w2_c)

			# Pad/trim rows to match module.out_features (ceiling-div)
			target_out = module.out_features
			if delta.shape[0] < target_out:
				delta = delta.repeat(
					(target_out + delta.shape[0] - 1) // delta.shape[0], 1)
			if delta.shape[0] > target_out:
				delta = delta[:target_out, :]

			# Pad/trim cols to match module.in_features
			target_in = module.in_features
			if delta.shape[1] < target_in:
				delta = delta.repeat(
					1, (target_in + delta.shape[1] - 1) // delta.shape[1])
			if delta.shape[1] > target_in:
				delta = delta[:, :target_in]

			# QuaRot: rotate expanded delta.
			# w2 col dim (64/16) rarely divides group_size (128),
			# but expanded delta columns (3840/10240/256) always do.
			if quarot_enabled and H is not None and delta.shape[1] % group_size == 0:
				delta = delta.to(dev)
				delta = _rot_quarot_tensor(delta, H, group_size, dev)
				delta = delta.to(cpu).contiguous().clone()
			else:
				delta = delta.contiguous().clone()

			object.__setattr__(module, '_tint4_lokr_kron_cache',
			                   {'key': cache_key, 'delta': delta})

		# Mirror ComfyUI LoKrAdapter: direct w1/w2 → scale=1.0
		mult = strength

		le = getattr(module, '_tint4_lora_entries', None)
		if le is None:
			le = {}
			object.__setattr__(module, '_tint4_lora_entries', le)

		if qkv_slice is not None:
			sl, se = qkv_slice
			delta_slice = delta[sl:se, :].contiguous().clone()
			le.setdefault(lora_name, []).append(
				("delta", delta_slice, mult, sl, se))
		else:
			le.setdefault(lora_name, []).append(
				("delta", delta, mult))


NODE_CLASS_MAPPINGS = {"TINT4LoRALoader": TINT4LoRALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TINT4LoRALoader": "TINT4 LoRA Loader"}
