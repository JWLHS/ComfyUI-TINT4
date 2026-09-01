"""
tint4_loader.py — TINT4 Model Loader v1.0 + v1.1 bypass fix

v1.1: load_model reads JS bypass signal + forces _tint4_reset_all_loras.
v8.5: +model_type_key(), +9 native detect functions, dropdown display-name format.
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
import gc, os, json, hashlib, math, time
import logging
import weakref
import folder_paths
import comfy.sd
import comfy.model_detection
import comfy.utils
import comfy.ops
import comfy.model_management
from torchao.quantization.quantize_.workflows.int4.int4_plain_int32_tensor import (
	Int4PlainInt32Tensor,
)
from safetensors import safe_open

log = logging.getLogger("TINT4-Loader")
log.info(f"[TINT4] TINT4_ONEDNN={os.environ.get('TINT4_ONEDNN', '<unset>')} (oneDNN u4 backend)")

# ---------------------------------------------------------------------------
# Unload integration: ComfyUI's unload_all_models() cannot see the GPU-cached
# int4 tensors (_qt) through AIMDO's detach wrapper, so VRAM stayed pinned
# after sampling (~10GiB for H3) and VAE decode OOM'd.  Register every loaded
# diffusion model here and flush its TINT4Linear caches whenever ComfyUI
# unloads all models.
# ---------------------------------------------------------------------------
_TINT4_MODEL_REFS: list = []

# ── qdata VRAM budget (LRU) ────────────────────────────────────────────
# Caching all 200 layers' int4 qdata (~10GiB) plus H3 conditioning leaves
# <2GiB of steady VRAM on a 16GiB card; step transients then sporadically
# OOM the L0 driver (error 40/20).  Cap the GPU-resident qdata and evict the
# least-recently-used layers back to CPU (re-upload ~0.2s/2GiB per step).
_TINT4_QDATA_BUDGET = 8 * 1024 ** 3
_TINT4_ONEDNN_BUDGET = 7.5 * 1024 ** 3
_TINT4_QDATA_LRU: list = []  # [(weakref to layer, size_bytes)]
_TINT4_EVICT_PENDING: list = []  # layers whose _qt must be freed AFTER a sync
# LRU 驱逐日志限频：每层 forward 都会 touch，逐次 info 会刷屏（H3 采样
# 时每步几十行）。累计驱逐层数，每 5 秒或每累计 100 层打一行汇总。
_TINT4_EVICT_ACC = 0
_TINT4_EVICT_LAST_LOG = 0.0


def _tint4_qdata_size(layer) -> int:
    try:
        n = 0
        qt = layer._qt
        if qt is not None and getattr(qt, "device", None) is not None and qt.device.type != "cpu":
            for attr in ("qdata", "scale", "zero_point"):
                t = getattr(qt, attr, None)
                if t is not None:
                    n += t.numel() * t.element_size()
        return n
    except Exception:
        return 0


def _tint4_touch(layer):
    try:
        # ── AIMDO 让路（2026-09-01，参照 int4xpu 正式版）──
        # DLL/劫持版 AIMDO 活跃时，权重不自己驱逐：常驻显存，显存整体
        # 水位（换 CLIP/VAE 等 staged 权重）交给 AIMDO 管。逐层驱逐重建
        # 是采样前停留 + CPU 打满的根源（H3 250 层每步驱逐/重建）。
        # 无 AIMDO 时保持原 LRU 行为。
        from .tint4_aimdo import is_aimdo_active
        if is_aimdo_active():
            return
        size = _tint4_qdata_size(layer)
        if size <= 0:
            return
        # move this layer to the MRU end
        _TINT4_QDATA_LRU[:] = [e for e in _TINT4_QDATA_LRU if e[0]() is not layer]
        _TINT4_QDATA_LRU.append((weakref.ref(layer), size))
        total = sum(s for _, s in _TINT4_QDATA_LRU)
        # oneDNN mode: keep the whole model resident (~10GB); the native u4
        # path rebuilds cached inputs per eviction, which is far costlier than
        # int4pack. 10GB resident + ~4GB sampling transients fits 15.56GB.
        budget = (_TINT4_ONEDNN_BUDGET
                  if os.environ.get("TINT4_ONEDNN") == "1"
                  else _TINT4_QDATA_BUDGET)
        evicted = 0
        while total > budget and len(_TINT4_QDATA_LRU) > 1:
            ref, sz = _TINT4_QDATA_LRU.pop(0)
            m = ref()
            if m is not None:
                if evicted == 0:
                    # One sync per eviction batch: guarantees every earlier
                    # layer's GEMM has finished, so freeing now is race-free
                    # and frees VRAM DURING the first forward (otherwise the
                    # run stays at the full ~10GiB and step transients OOM the
                    # L0 driver with only ~0.2GiB free).
                    try:
                        torch.xpu.synchronize()
                    except Exception:
                        pass
                if m._qt is not None and m._qt.device.type != "cpu":
                    m._qt = None
                m._onednn_packed = None
                m._onednn_scales = None
                m._onednn_corr = None
            total -= sz
            evicted += 1
        if evicted:
            global _TINT4_EVICT_ACC, _TINT4_EVICT_LAST_LOG
            _TINT4_EVICT_ACC += evicted
            _now = time.time()
            if _now - _TINT4_EVICT_LAST_LOG >= 5.0 or _TINT4_EVICT_ACC >= 100:
                log.info(
                    f"[TINT4] qdata LRU evicted {_TINT4_EVICT_ACC} layers "
                    f"(resident {total/1e9:.1f}GB)"
                )
                _TINT4_EVICT_ACC = 0
                _TINT4_EVICT_LAST_LOG = _now
    except Exception:
        pass


def _tint4_flush_pending():
    """Free deferred evictions.  Caller must have synchronized the XPU queue."""
    try:
        for m in _TINT4_EVICT_PENDING:
            if m._qt is not None and m._qt.device.type != "cpu":
                m._qt = None
        _TINT4_EVICT_PENDING.clear()
    except Exception:
        pass

def _register_tint4_model(model):
    try:
        _ensure_unload_hook()
        dm = model.model.diffusion_model
        while hasattr(dm, "_orig_mod"):
            dm = dm._orig_mod
        _TINT4_MODEL_REFS.append(weakref.ref(dm))
    except Exception:
        pass

def _flush_tint4_all():
    dead = []
    for ref in _TINT4_MODEL_REFS:
        dm = ref()
        if dm is None:
            dead.append(ref)
            continue
        try:
            for m in dm.modules():
                if isinstance(m, TINT4Linear):
                    m.release_xpu()
        except Exception:
            pass
    for d in dead:
        _TINT4_MODEL_REFS.remove(d)
    if torch.xpu.is_available():
        # AIMDO 让路：活跃时不主动清缓存（水位由 AIMDO 管）
        try:
            from .tint4_aimdo import is_aimdo_active
            if not is_aimdo_active():
                torch.xpu.empty_cache()
        except Exception:
            torch.xpu.empty_cache()

_orig_unload_all_models = comfy.model_management.unload_all_models
def _unload_all_models_with_tint4():
    _flush_tint4_all()
    return _orig_unload_all_models()

def _ensure_unload_hook():
    # Re-assert the wrapper at every load in case another plugin (AIMDO)
    # replaced unload_all_models after we were imported.
    global _orig_unload_all_models
    if comfy.model_management.unload_all_models is not _unload_all_models_with_tint4:
        _orig_unload_all_models = comfy.model_management.unload_all_models
        comfy.model_management.unload_all_models = _unload_all_models_with_tint4

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
		# Registered (non-persistent) buffer so ComfyUI's model unload/offload
		# machinery moves the GPU-cached int4 tensor back to CPU and frees VRAM.
		# Before this, _qt lived in a plain attribute (~10GiB for H3) that
		# unload_all_models() could not see -> VRAM stayed pinned after sampling.
		self.register_buffer("_qt", None, persistent=False)
		self._use_quarot: bool = False
		self._group_size: int = 128
		self._hadamard_H = None
		self._tint4_lora_entries: dict | None = None
		self._tint4_bake_state: dict | None = None
		# oneDNN u4 GEMM backend: avoids the unstable torch int4pack op on XPU
		# (M=237 driver crash) and uses Intel's native u4 matmul. Enable with
		# TINT4_ONEDNN=1.
		self._onednn_packed: torch.Tensor | None = None
		self._onednn_scales: torch.Tensor | None = None
		self._onednn_corr: torch.Tensor | None = None
		self._use_onednn: bool = os.environ.get("TINT4_ONEDNN", "0") == "1"

	def __del__(self):
		try:
			self._qdata = None
			self._scale = None
			self._zp = None
			self._qt = None
			self._tint4_lora_entries = None
			self._tint4_bake_state = None
		except Exception:
			pass

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
		self._onednn_packed = None
		self._onednn_scales = None
		self._onednn_corr = None

	def _apply(self, fn, *args, **kwargs):
		# _qt 注册为 buffer（便于 ComfyUI unload/offload 机制感知），但它是
		# torchao Int4PlainInt32Tensor 子类：wan 块换入换出走 module.to()
		# 时 torchao dispatch 的 storage 别名修正会跨设备崩溃
		# （"Attempted to set the storage ... different device"）。
		# 设备迁移前先失效设备缓存，_qt 置 None 后 _apply 自动跳过；
		# 下次 forward 会按需从 _qdata 重建。
		self._qt = None
		self._onednn_packed = None
		self._onednn_scales = None
		self._onednn_corr = None
		return super()._apply(fn, *args, **kwargs)

	def _get_onednn_inputs(self, dev):
		"""Lazily convert torchao plain_int32 qdata -> oneDNN u4 inputs.

		plain_int32 layout: each int32 holds 8 int4 values (4 bytes x 2 nibbles,
		low nibble = even column).  Viewed as uint8 this is exactly the oneDNN
		[N, K/2] u4 layout (byte j = columns 2j, 2j+1, low nibble first).
		oneDNN applies a fixed scalar zp=8 ((u4-8)*scale); torchao asymmetric
		quantization uses a per-group zp, so we add the correction
		out += rowsum_per_group(act) @ corr^T  with corr = (8 - zp) * scale.
		"""
		if self._onednn_packed is not None and self._onednn_packed.device == dev:
			return self._onednn_packed, self._onednn_scales, self._onednn_corr
		if not getattr(self, "_onednn_diag", False) and os.environ.get("TINT4_ONEDNN_DEBUG") == "1":
			object.__setattr__(self, "_onednn_diag", True)
			log.info(f"[TINT4] onednn first build layer {self.out_features}x{self.in_features}")
		qb = self._qdata.view(torch.uint8).reshape(
			self.out_features, self.in_features // 2).contiguous()
		s = self._scale
		z = self._zp
		packed = qb.to(dev)
		# TINT4 stores scale/zp as [G, N] (groups x out_features).
		scales = s.float().contiguous().to(torch.float16).to(dev)
		corr = ((8.0 - z.float()) * s.float()).contiguous().to(torch.float16).to(dev)
		self._onednn_packed = packed
		self._onednn_scales = scales
		self._onednn_corr = corr
		if os.environ.get("TINT4_ONEDNN_DEBUG") == "1":
			log.info(f"[TINT4] onednn inputs diag {self.out_features}x{self.in_features}: "
					f"qdata_nan={int(torch.isnan(self._qdata.float()).sum().item())} "
					f"scale_nan={int(torch.isnan(self._scale.float()).sum().item())} "
					f"zp_min={float(self._zp.float().min().item())} zp_max={float(self._zp.float().max().item())} "
					f"scale_min={float(self._scale.float().min().item())} scale_max={float(self._scale.float().max().item())} "
					f"scale_inf={int(torch.isinf(self._scale.float()).sum().item())} "
					f"quarot={self._use_quarot}")
		return packed, scales, corr

	def _dequant_fp16(self, dev):
		"""Reconstruct fp16 weights from the plain_int32 layout (CPU tensors).

		Packing (torchao 'plain_int32' / 'n'): each int32 holds 4 little-endian
		bytes, each byte packs 2 int4 values (even column in the low nibble).
		Used for small-M forwards where torch's XPU int4pack GEMM is unstable
		(e.g. M=237 hard-crashes the L0 driver even standalone).
		"""
		bs = self._block_size
		gs = bs[1] if isinstance(bs, (tuple, list)) else bs
		q = self._qdata
		s = self._scale
		z = self._zp
		out_f, k8 = q.shape
		k = k8 * 8
		qb = q.view(torch.uint8).reshape(out_f, k8, 4)
		lo = (qb & 0x0F).to(torch.float16)
		hi = ((qb >> 4) & 0x0F).to(torch.float16)
		vals = torch.stack([lo, hi], dim=-1).reshape(out_f, k)
		idx = torch.arange(k, device=q.device) // gs
		sc = s.t().to(torch.float16)[:, idx]
		zc = z.t().to(torch.float16)[:, idx]
		return ((vals - zc) * sc).to(dev)

	def forward(self, x):
		x2 = x.reshape(-1, x.shape[-1])
		if not getattr(self, "_dtype_diag", False) and os.environ.get("TINT4_ONEDNN_DEBUG") == "1":
			object.__setattr__(self, "_dtype_diag", True)
			log.info(f"[TINT4] forward x dtype={x.dtype} x2.shape={tuple(x2.shape)}")
		if self._use_quarot and self._hadamard_H is not None:
			try:
				from .wint8_quarot import rotate_activation
				x2 = rotate_activation(x2, self._hadamard_H, self._group_size)
			except Exception:
				pass
		dev = x.device
		if x2.shape[0] < 512:
			# Small forwards (conditioning/ref passes): torch's XPU int4pack
			# GEMM is unstable for some M (M=237 hard-crashes the driver);
			# use a plain fp16 GEMM instead.  Peak cost is negligible at
			# these sizes.
			# 反量化权重必须对齐激活 dtype：fp16 模型走 fp16（行为不变），
			# bf16 模型（如 MiniMax H3 全量底模）走 bf16，否则 bf16×fp16
			# dtype 不匹配崩溃（实测 adaln_proj 输入为 bf16）。
			out = F.linear(x2, self._dequant_fp16(dev).to(x2.dtype), None)
		else:
			# oneDNN u4 GEMM is numerically unstable at very large M in this
			# driver/oneDNN combo (intermittent all-NaN after several calls);
			# keep the big sampling forwards on the proven int4pack path.
			if self._use_onednn and x2.shape[0] < 4096:
				try:
					from omni_xpu_kernel import svdq as _svdq
					packed, scales, corr = self._get_onednn_inputs(dev)
					x2c = x2.contiguous()
					# oneDNN bf16 u4 GEMM is numerically poor on large M
					# (measured ~2.5% rel error vs 0.28% for f16) -> compute in
					# f16, then cast back to the model dtype.
					act_f16 = x2c.to(torch.float16) if x2c.dtype != torch.float16 else x2c
					out = _svdq.onednn_int4_gemm_preconverted(act_f16, packed, scales)
					num_groups = scales.shape[0]
					gs = self.in_features // num_groups
					act_gs = act_f16.reshape(act_f16.shape[0], num_groups, gs).sum(dim=-1)
					out = out + act_gs @ corr
					if x2c.dtype != torch.float16:
						out = out.to(x2c.dtype)
					_tint4_touch(self)
					if not getattr(self, "_nan_diag", False) and os.environ.get("TINT4_ONEDNN_DEBUG") == "1":
						_nan = int(torch.isnan(out.float()).sum().item())
						_inf = int(torch.isinf(out.float()).sum().item())
						_absmax = float(out.abs().max().item()) if out.numel() else 0.0
						_mean = float(out.float().mean().item()) if out.numel() else 0.0
						object.__setattr__(self, "_nan_diag", True)
						log.info(f"[TINT4] onednn out layer {self.out_features}x{self.in_features} "
								f"nan={_nan} inf={_inf} absmax={_absmax:.4f} mean={_mean:.6f}")
				except Exception as e:
					log.warning(f"[TINT4] oneDNN GEMM failed, falling back to int4pack: {e}")
					self._use_onednn = False
					if self._qt is None or self._qt.device != dev:
						self._qt = Int4PlainInt32Tensor(
							self._qdata.to(dev), self._scale.to(dev), self._zp.to(dev),
							self._block_size, [self.out_features, self.in_features],
						)
					_tint4_touch(self)
					out = F.linear(x2, self._qt, None)
			else:
				if self._qt is None or self._qt.device != dev:
					self._qt = Int4PlainInt32Tensor(
						self._qdata.to(dev), self._scale.to(dev), self._zp.to(dev),
						self._block_size, [self.out_features, self.in_features],
					)
				_tint4_touch(self)
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
					if isinstance(e[0], str) and e[0] == "delta":
							_, delta_cpu, mult = e[:3]
							sl = e[3] if len(e) > 3 else None
							se = e[4] if len(e) > 4 else None
							delta_gpu = delta_cpu.to(device=dev, dtype=cd)
							lo = x2.to(cd) @ delta_gpu.T * mult
							if sl is not None:
								out[:, sl:se] += lo
							else:
								out += lo
							continue
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


# ── 已有自定义检测 ────────────────────────────────────────────────

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


# ── 新增自定义检测（v8.5）— 只用 _EXCLUSIONS 中的排除层 key ──────

def _detect_sd3(sd: dict):
	"""SD3 / SD3.5 MMDiT — 排除层: x_embedder, y_embedder, context_embedder, final_layer, pos_embed"""
	x_key = "x_embedder.proj.weight"
	if x_key not in sd:
		return None
	cfg = {}
	cfg["image_model"] = "sd3"
	cfg["in_channels"] = sd[x_key].shape[1]
	cfg["patch_size"] = sd[x_key].shape[2]
	cfg["depth"] = sd[x_key].shape[0] // 64
	cfg["input_size"] = None
	fl_key = "final_layer.linear.weight"
	if fl_key in sd:
		cfg["out_channels"] = sd[fl_key].shape[0] // (cfg["patch_size"] ** 2)
	y_key = "y_embedder.mlp.0.weight"
	if y_key in sd:
		cfg["adm_in_channels"] = sd[y_key].shape[1]
	ctx_key = "context_embedder.weight"
	if ctx_key in sd:
		cfg["context_embedder_config"] = {
			"target": "torch.nn.Linear",
			"params": {
				"in_features": sd[ctx_key].shape[1],
				"out_features": sd[ctx_key].shape[0],
			},
		}
	pe_key = "pos_embed"
	if pe_key in sd:
		cfg["num_patches"] = sd[pe_key].shape[1]
		cfg["pos_embed_max_size"] = round(math.sqrt(sd[pe_key].shape[1]))
	cfg["pos_embed_scaling_factor"] = None
	cfg["qk_norm"] = None
	cfg["x_block_self_attn_layers"] = []
	return cfg


def _detect_flux(sd: dict):
	"""Flux.1 MMDiT — 排除层: img_in, txt_in, time_in, vector_in, guidance_in, final_layer, img_mod.lin, txt_mod.lin, modulation.lin"""
	img_key = "img_in.weight"
	if img_key not in sd:
		return None
	cfg = {}
	cfg["image_model"] = "flux"
	cfg["hidden_size"] = sd[img_key].shape[0]
	cfg["in_channels"] = sd[img_key].shape[1]
	txt_key = "txt_in.weight"
	if txt_key in sd:
		cfg["context_in_dim"] = sd[txt_key].shape[1]
	vec_key = "vector_in.in_layer.weight"
	cfg["vec_in_dim"] = sd[vec_key].shape[1] if vec_key in sd else None
	cfg["guidance_embed"] = "guidance_in.in_layer.weight" in sd
	cfg["in_dim"] = cfg["in_channels"]
	cfg["out_dim"] = cfg["hidden_size"]
	cfg["axes_dim"] = [16, 24, 24]
	cfg["theta"] = 10000.0
	cfg["patch_size"] = 1
	cfg["out_channels"] = cfg["hidden_size"]
	cfg["num_heads"] = cfg["hidden_size"] // sum(cfg["axes_dim"])
	cfg["depth"] = comfy.model_detection.count_blocks(
		list(sd.keys()), "double_blocks." + "{}.")
	cfg["depth_single_blocks"] = comfy.model_detection.count_blocks(
		list(sd.keys()), "single_blocks." + "{}.")
	cfg["guidance_embed"] = "guidance_in.in_layer.weight" in sd
	cfg["yak_mlp"] = False
	cfg["txt_ids_dims"] = [3]
	return cfg


def _detect_auraflow(sd: dict):
	"""AuraFlow DiT — 排除层: positional_encoding, cond_seq_linear"""
	pe_key = "positional_encoding"
	cond_key = "cond_seq_linear.weight"
	if pe_key not in sd or cond_key not in sd:
		return None
	cfg = {}
	cfg["image_model"] = "auraflow"
	cfg["max_seq"] = sd[pe_key].shape[1]
	cfg["cond_seq_dim"] = sd[cond_key].shape[1]
	keys = list(sd.keys())
	cfg["n_double_layers"] = comfy.model_detection.count_blocks(
		keys, "double_layers." + "{}.")
	cfg["n_layers"] = (cfg["n_double_layers"]
		+ comfy.model_detection.count_blocks(keys, "single_layers." + "{}."))
	return cfg


def _detect_cosmos(sd: dict):
	"""Cosmos — 排除层: x_embedder, final_layer, adaln, t_embedder"""
	x_key = "x_embedder.proj.1.weight"
	if x_key not in sd:
		return None
	model_channels = sd[x_key].shape[0]
	in_channels = (sd[x_key].shape[1] // 4) - 1

	cfg = {
		"image_model": "cosmos",
		"model_channels": model_channels,
		"max_img_h": 240, "max_img_w": 240, "max_frames": 128,
		"in_channels": in_channels,
		"out_channels": 16,
		"patch_spatial": 2, "patch_temporal": 1,
		"block_config": "FA-CA-MLP",
		"concat_padding_mask": True,
		"pos_emb_cls": "rope3d",
		"pos_emb_learnable": False,
		"pos_emb_interpolation": "crop",
		"block_x_format": "THWBD",
		"affline_emb_norm": True,
		"use_adaln_lora": True, "adaln_lora_dim": 256,
	}

	if model_channels == 4096:
		cfg["num_blocks"] = 28
		cfg["num_heads"] = 32
		cfg["extra_per_block_abs_pos_emb"] = True
		cfg["rope_h_extrapolation_ratio"] = 1.0
		cfg["rope_w_extrapolation_ratio"] = 1.0
		cfg["rope_t_extrapolation_ratio"] = 2.0
		cfg["extra_per_block_abs_pos_emb_type"] = "learnable"
	else:
		cfg["num_blocks"] = 36
		cfg["num_heads"] = 40
		cfg["extra_per_block_abs_pos_emb"] = True
		cfg["rope_h_extrapolation_ratio"] = 2.0
		cfg["rope_w_extrapolation_ratio"] = 2.0
		cfg["rope_t_extrapolation_ratio"] = 2.0
		cfg["extra_h_extrapolation_ratio"] = 2.0
		cfg["extra_w_extrapolation_ratio"] = 2.0
		cfg["extra_t_extrapolation_ratio"] = 2.0
		cfg["extra_per_block_abs_pos_emb_type"] = "learnable"

	log.info(f"[TINT4] cosmos detected: model_channels={model_channels}, "
			 f"in_channels={in_channels}, num_blocks={cfg['num_blocks']}")
	return cfg


def _detect_cogvideox(sd: dict):
	"""CogVideoX DiT — 排除层: patch_embed, proj_out, ofs_embedding"""
	pe_key = "patch_embed.proj.weight"
	if pe_key not in sd:
		return None
	cfg = {}
	cfg["image_model"] = "cogvideox"
	po_key = "proj_out.weight"
	if po_key in sd:
		cfg["out_channels"] = sd[po_key].shape[0] // 4
	ofs_key = "ofs_embedding_linear_1.weight"
	if ofs_key in sd:
		cfg["ofs_embed_dim"] = sd[ofs_key].shape[1]
	return cfg


def _detect_lens(sd: dict):
	"""Lens DiT — 排除层: img_in, proj_out, txt_norm"""
	img_key = "img_in.weight"
	if img_key not in sd:
		return None
	proj_key = "proj_out.weight"
	if proj_key not in sd:
		return None
	cfg = {}
	cfg["image_model"] = "lens"
	cfg["in_channels"] = sd[img_key].shape[1]
	cfg["out_channels"] = sd[proj_key].shape[0] // 4
	cfg["num_layers"] = comfy.model_detection.count_blocks(
		list(sd.keys()), "transformer_blocks." + "{}.")
	cfg["num_attention_heads"] = sd[img_key].shape[0] // 64
	multi_layer = "txt_norm.0.weight" in sd
	if multi_layer:
		cfg["enc_hidden_dim"] = sd["txt_norm.0.weight"].shape[0]
		cfg["selected_layer_index"] = tuple(range(
			comfy.model_detection.count_blocks(list(sd.keys()), "txt_norm." + "{}.")))
	else:
		cfg["enc_hidden_dim"] = sd["txt_norm.weight"].shape[0]
		cfg["selected_layer_index"] = (0,)
	cfg["multi_layer_encoder_feature"] = multi_layer
	return cfg


def _detect_kandinsky5(sd: dict):
	"""Kandinsky 5 — 排除层: visual_embeddings, time_embeddings"""
	ve_key = "visual_embeddings.in_layer.bias"
	if ve_key not in sd:
		return None
	model_dim = sd[ve_key].shape[0]
	cfg = {}
	cfg["image_model"] = "kandinsky5"
	cfg["model_dim"] = model_dim
	if model_dim in [4096, 2560]:
		cfg["axes_dims"] = (32, 48, 48)
	elif model_dim == 1792:
		cfg["axes_dims"] = (16, 24, 24)
	te_key = "time_embeddings.in_layer.bias"
	cfg["time_dim"] = sd[te_key].shape[0] if te_key in sd else 0
	cfg["num_text_blocks"] = comfy.model_detection.count_blocks(
		list(sd.keys()), "text_transformer_blocks." + "{}.")
	cfg["num_visual_blocks"] = comfy.model_detection.count_blocks(
		list(sd.keys()), "visual_transformer_blocks." + "{}.")
	return cfg


def _detect_seedvr2(sd: dict):
	"""SeedVR 2 — 排除层: x_embedder, final_layer"""
	x_key = "x_embedder.weight"
	if x_key not in sd:
		return None
	cfg = {}
	cfg["image_model"] = "seedvr2"
	fl_key = "final_layer.linear.weight"
	if fl_key in sd:
		cfg["out_channels"] = sd[fl_key].shape[0] // 4
	return cfg


def _detect_anima(sd: dict):
	"""Anima — 排除层: adaln, x_embedder"""
	x_key = "x_embedder.proj.1.weight"
	if x_key not in sd:
		return None
	model_channels = sd[x_key].shape[0]
	in_channels = (sd[x_key].shape[1] // 4) - 1

	if model_channels == 2048:
		num_blocks, num_heads = 28, 16
	elif model_channels == 5120:
		num_blocks, num_heads = 36, 40
	else:
		return None

	cfg = {
		"image_model": "anima",
		"model_channels": model_channels,
		"num_blocks": num_blocks,
		"num_heads": num_heads,
		"max_img_h": 240, "max_img_w": 240, "max_frames": 128,
		"in_channels": in_channels,
		"out_channels": 16,
		"patch_spatial": 2, "patch_temporal": 1,
		"crossattn_emb_channels": 1024,
		"pos_emb_cls": "rope3d",
		"pos_emb_learnable": True,
		"pos_emb_interpolation": "crop",
		"min_fps": 1, "max_fps": 30,
		"use_adaln_lora": True, "adaln_lora_dim": 256,
		"concat_padding_mask": True,
	}

	if in_channels == 16:
		cfg["extra_per_block_abs_pos_emb"] = False
		cfg["rope_h_extrapolation_ratio"] = 4.0
		cfg["rope_w_extrapolation_ratio"] = 4.0
		cfg["rope_t_extrapolation_ratio"] = 1.0
	elif in_channels == 17:
		if model_channels == 2048:
			cfg["extra_per_block_abs_pos_emb"] = False
			cfg["rope_h_extrapolation_ratio"] = 3.0
			cfg["rope_w_extrapolation_ratio"] = 3.0
			cfg["rope_t_extrapolation_ratio"] = 1.0
		elif model_channels == 5120:
			cfg["rope_h_extrapolation_ratio"] = 2.0
			cfg["rope_w_extrapolation_ratio"] = 2.0
			cfg["rope_t_extrapolation_ratio"] = 0.8333333333333334

	cfg["extra_h_extrapolation_ratio"] = 1.0
	cfg["extra_w_extrapolation_ratio"] = 1.0
	cfg["extra_t_extrapolation_ratio"] = 1.0
	cfg["rope_enable_fps_modulation"] = False

	log.info(f"[TINT4] anima detected: model_channels={model_channels}, "
			 f"in_channels={in_channels}, num_blocks={num_blocks}")
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
	# v8.5 — 9 个新增自定义检测（检测入口在量化层，但排除层可获取参数）
	if model_type == "sd3":
		cfg = _detect_sd3(sd)
		if cfg is not None:
			return cfg
	if model_type == "flux":
		cfg = _detect_flux(sd)
		if cfg is not None:
			return cfg
	if model_type == "auraflow":
		cfg = _detect_auraflow(sd)
		if cfg is not None:
			return cfg
	if model_type == "cosmos":
		cfg = _detect_cosmos(sd)
		if cfg is not None:
			return cfg
	if model_type == "cogvideox":
		cfg = _detect_cogvideox(sd)
		if cfg is not None:
			return cfg
	if model_type == "lens":
		cfg = _detect_lens(sd)
		if cfg is not None:
			return cfg
	if model_type == "kandinsky5":
		cfg = _detect_kandinsky5(sd)
		if cfg is not None:
			return cfg
	if model_type == "seedvr2":
		cfg = _detect_seedvr2(sd)
		if cfg is not None:
			return cfg
	if model_type == "anima":
		cfg = _detect_anima(sd)
		if cfg is not None:
			return cfg
	if model_type == "minimax_h3":
		# MiniMax H3：ComfyUI 原生检测（video_patch_proj + audio_patch_proj），
		# 排除列表与 comfy-kitchen INT4_CONVROT 保持一致，检测逻辑复用官方实现
		return _orig_detect(sd, key_prefix, metadata)
	# 其余模型（检测入口在排除层中）走 _orig_detect 正常工作
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
					{"default": "flux2 (Flux.2)",
					 "tooltip": "Must match quantization type"},
				),
			},
		}

	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_model"

	def load_model(self, unet_name, model_type):
		from .tint4_quantizer import model_type_key
		model_type = model_type_key(model_type)

		# TINT4's real GPU footprint (~10-11GiB once _qt uploads) is invisible
		# to ComfyUI's planner until the first forward, so a previous run's VAE /
		# other models can still be resident and the qdata upload OOMs (error 40)
		# or kills the L0 context.  Unload everything first for a clean slate.
		try:
			comfy.model_management.unload_all_models()
			from .tint4_aimdo import is_aimdo_active
			if not is_aimdo_active():
				comfy.model_management.soft_empty_cache()
			log.info("[TINT4] Pre-load: unloaded all other models (clean VRAM)")
		except Exception:
			pass

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

		# 占位符注入必须在模型构建之前无条件执行（首次加载无缓存文件时
		# 同样需要），否则 sd 缺少量化层权重，ComfyUI 构建会 KeyError。
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

		# ── 缺失权重占位瘦身（同 tint4_loader_ltx）─────────────────
		# 量化层权重已进 quant_map，占位符只给少数检测键；其余缺失键在
		# disable_weight_init 懒加载里会补全尺寸 zeros（H3 实测加载峰值
		# 52GB）。量化层随后被 TINT4Linear 注入替换，大参数缺失层放 1x1
		# 占位即可，注入后即释放。
		from comfy.ops import disable_weight_init as _dwi
		_orig_lazy = _dwi._lazy_load_from_state_dict

		def _tiny_lazy_load(module, state_dict, prefix, local_metadata,
							missing_keys, unexpected_keys, weight_shape,
							bias_shape=None):
			assign_to_params_buffers = local_metadata.get(
				"assign_to_params_buffers", False)
			prefix_len = len(prefix)
			for k, v in state_dict.items():
				key = k[prefix_len:]
				if key == "weight":
					if not assign_to_params_buffers:
						v = v.clone()
					module.weight = torch.nn.Parameter(v, requires_grad=False)
				elif bias_shape is not None and key == "bias" and v is not None:
					if not assign_to_params_buffers:
						v = v.clone()
					module.bias = torch.nn.Parameter(v, requires_grad=False)
				else:
					unexpected_keys.append(k)
			if module.weight is None:
				_params = weight_shape[0] * weight_shape[1]
				if _params >= 1024 * 1024:
					module.weight = torch.nn.Parameter(
						torch.zeros((1, 1), dtype=torch.float16),
						requires_grad=False)
				else:
					module.weight = torch.nn.Parameter(
						torch.zeros(weight_shape), requires_grad=False)
				missing_keys.append(prefix + "weight")
			if (bias_shape is not None and module.bias is None
					and getattr(module, "comfy_need_lazy_init_bias", False)):
				module.bias = torch.nn.Parameter(
					torch.zeros(bias_shape), requires_grad=False)
				missing_keys.append(prefix + "bias")

		_dwi._lazy_load_from_state_dict = staticmethod(_tiny_lazy_load)
		comfy.model_detection.detect_unet_config = _detect_wrapper
		try:
			model = comfy.sd.load_diffusion_model_state_dict(
				sd, model_options={
					"custom_operations": comfy.ops.manual_cast,
				}, metadata={})
		finally:
			comfy.model_detection.detect_unet_config = _orig_detect
			_dwi._lazy_load_from_state_dict = _orig_lazy

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
						  f"model.{full}", 
						  f"net.{full}", 
						  full]:
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

		_tint4_reset_all_loras(model)
		if force_reset:
			log.info("[TINT4] ✓ LoRA state force-cleared after model load")

		log.info(
			f"[TINT4] Loaded '{unet_name}' | {injected} INT4 layers"
		)
		try:
			import comfy.ldm.minimax.model as _h3m
			_h3m._tint4_flush_hook = _tint4_flush_pending
		except Exception:
			pass
		_register_tint4_model(model)
		return (model,)


NODE_CLASS_MAPPINGS = {"TINT4ModelLoader": TINT4ModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TINT4ModelLoader": "TINT4 Model Loader"}
