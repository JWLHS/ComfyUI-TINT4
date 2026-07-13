"""tint4_aimdo.py — TINT4 × AIMDO bridge v1.1

Exports:
    is_aimdo_active()        → bool
    build_weight_placeholders(quant_specs, sd) → (count, log_suffix)
    register_aimdo_hooks(diffusion_model)      → None
    patch_model_for_aimdo(model)               → None
"""
import logging
import torch
import gc

log = logging.getLogger("TINT4-AIMDO")

_aimdo_ctrl = None
_has_aimdo = False

_aimdo_cuda_sync = torch.cuda.synchronize
_noop_sync = lambda: None


def _check_aimdo():
    global _aimdo_ctrl, _has_aimdo
    import sys
    mod = sys.modules.get("comfy_aimdo")
    if mod is None:
        _has_aimdo = False
        _aimdo_ctrl = None
        return
    try:
        _aimdo_ctrl = mod.control
        _has_aimdo = True
    except Exception:
        _has_aimdo = False
        _aimdo_ctrl = None


def _restore_cuda_sync():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.cuda.synchronize = _noop_sync


def is_aimdo_active() -> bool:
    _check_aimdo()
    if not _has_aimdo or _aimdo_ctrl is None:
        _restore_cuda_sync()
        return False
    try:
        act = getattr(_aimdo_ctrl, '_dynamic_vram_enabled', False)
        if act:
            torch.cuda.synchronize = _aimdo_cuda_sync
        else:
            _restore_cuda_sync()
        return act
    except Exception:
        _restore_cuda_sync()
        return False


def build_weight_placeholders(quant_specs, sd):
    if is_aimdo_active():
        for _base, _sh0, _sh1 in quant_specs:
            sd[f"{_base}.weight"] = torch.zeros(_sh0, _sh1, dtype=torch.uint8)
        return len(quant_specs), "(AIMDO mode — full placeholders)"
    else:
        _specs_sorted = sorted(quant_specs, key=lambda x: x[0])
        _seen = set()
        _chosen = []
        for _base, _sh0, _sh1 in _specs_sorted:
            _parts = _base.split(".")
            _pfx = ".".join(_parts[:2]) if len(_parts) >= 2 else _base
            if _pfx not in _seen:
                _seen.add(_pfx)
                _chosen.append((_base, _sh0, _sh1))
            if len(_chosen) >= 6:
                break
        for _base, _sh0, _sh1 in _chosen:
            sd[f"{_base}.weight"] = torch.zeros(_sh0, _sh1, dtype=torch.uint8)
        return len(_chosen), "(standard mode — minimal placeholders)"


def register_aimdo_hooks(diffusion_model):
    pass


def _flush_tint4_caches(diffusion_model):
    from .tint4_loader import TINT4Linear
    for m in diffusion_model.modules():
        if isinstance(m, TINT4Linear):
            m.release_xpu()


def patch_model_for_aimdo(model):
    _check_aimdo()
    if not _has_aimdo:
        return

    from comfy_aimdo import control as _ctrl
    from .tint4_lora_common import _tint4_reset_all_loras, _empty_accelerator_cache

    dm = model.model.diffusion_model
    while hasattr(dm, '_orig_mod'):
        dm = dm._orig_mod

    register_aimdo_hooks(dm)

    _orig_detach = model.detach
    def _aimdo_detach(unpatch_all=True):
        _tint4_reset_all_loras(model)
        _dm = model.model.diffusion_model
        while hasattr(_dm, '_orig_mod'):
            _dm = _dm._orig_mod
        _flush_tint4_caches(_dm)
        _empty_accelerator_cache()
        gc.collect()
        return _orig_detach(unpatch_all)

    object.__setattr__(model, 'detach', _aimdo_detach)
    log.info("[TINT4 AIMDO] Model patched for AIMDO (detach wrapper active)")
