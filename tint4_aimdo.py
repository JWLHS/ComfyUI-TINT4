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


def aimdo_state() -> str:
    """运行时 AIMDO 管控三态：
    'none'   = 无 AIMDO 管控（未装 / cuda 版不生效）
    'hijack' = 劫持版（纯 Python，无 DLL；_dynamic_vram_enabled 标记）
    'dll'    = DLL XPU 版（aimdo_xpu.dll；lib 存在 + implementation=xpu
               + _xpu_allocator_ready）
    'broken' = DLL 加载但设备未初始化
    """
    _check_aimdo()
    if not _has_aimdo or _aimdo_ctrl is None:
        return "none"
    try:
        ctrl = _aimdo_ctrl
        # DLL XPU 版：lib 存在 + implementation == xpu
        if (getattr(ctrl, "lib", None) is not None
                and getattr(ctrl, "implementation", None) == "xpu"):
            if getattr(ctrl, "_xpu_allocator_ready", False):
                return "dll"
            return "broken"
        # 劫持版：无 DLL（lib None），_dynamic_vram_enabled 标记
        hijack = getattr(ctrl, "_dynamic_vram_enabled", False)
        if not hijack:
            try:
                hijack = callable(getattr(ctrl, "is_dynamic_vram_enabled", None)) \
                    and bool(ctrl.is_dynamic_vram_enabled())
            except Exception:
                hijack = False
        if hijack:
            return "hijack"
        # comfy 兜底：aimdo_enabled 无法细分时按劫持处理
        try:
            import comfy.memory_management as _mm
            if getattr(_mm, "aimdo_enabled", False):
                return "hijack"
        except Exception:
            pass
        return "none"
    except Exception:
        return "none"


def is_aimdo_active() -> bool:
    """统一查询：AIMDO 是否活跃（DLL 或劫持版都算）"""
    st = aimdo_state()
    act = st in ("hijack", "dll")
    try:
        if act:
            # 仅 CUDA/ROCm 后端需要真实 cuda.synchronize；
            # XPU 构建 torch.cuda.synchronize() 直接抛
            # "Torch not compiled with CUDA enabled"，必须保持 noop。
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                torch.cuda.synchronize = _aimdo_cuda_sync
        else:
            _restore_cuda_sync()
    except Exception:
        _restore_cuda_sync()
    return act


def build_weight_placeholders(quant_specs, sd):
    # 占位符只服务两个目的：1) 让 comfy 构建模型时不因缺失权重 KeyError；
    # 2) 让架构检测能读到少数关键层的真实形状（H3 的 blocks.0.attn.qkv_proj
    # / blocks.0.mlp.fc1）。量化层构建后立即被 TINT4Linear 注入替换，其余层
    # 无需全尺寸占位。此前 AIMDO 分支给全部量化层建全尺寸 uint8 占位，H3
    # （~30B 参数）加载时临时多占 ~30GB RAM（实测峰值 52GB）；改为最小占位
    # 后加载期 RAM 恢复正常（与 wan/tint4-ltx 的 1x1 占位瘦身一致）。
    _specs_sorted = sorted(quant_specs, key=lambda x: x[0])
    # MiniMax H3：ComfyUI 架构检测会直接读取
    # blocks.0.attn.qkv_proj.weight / blocks.0.mlp.fc1.weight，
    # 而 H3 所有层前缀都是 blocks.0，按前缀去重会漏掉它们。
    # 这里强制把这两个检测键纳入占位符。
    _chosen = []
    for _base, _sh0, _sh1 in _specs_sorted:
        if _base in ("blocks.0.attn.qkv_proj", "blocks.0.mlp.fc1"):
            _chosen.append((_base, _sh0, _sh1))
    _seen = set()
    # Qwen Image 2.1：ComfyUI 的架构检测要求 transformer_blocks.0.img_mlp
    # .{gate_up,proj}.weight 存在（只读 shape[0]），而它们都是量化层、会被
    # 前缀去重挤掉 → 检测返回 None → 建模型失败（实测 'NoneType' object
    # has no attribute 'model'）。这里强制纳入占位符（同 H3 的处理）。
    _qwen_detect = ("transformer_blocks.0.img_mlp.gate_up",
                    "transformer_blocks.0.img_mlp.proj")
    for _base, _sh0, _sh1 in _specs_sorted:
        if _base in _qwen_detect:
            _chosen.append((_base, _sh0, _sh1))
    for _base, _sh0, _sh1 in _specs_sorted:
        if (_base, _sh0, _sh1) in _chosen:
            continue
        _parts = _base.split(".")
        _pfx = ".".join(_parts[:2]) if len(_parts) >= 2 else _base
        if _pfx not in _seen:
            _seen.add(_pfx)
            _chosen.append((_base, _sh0, _sh1))
        if len(_chosen) >= 6:
            break
    for _base, _sh0, _sh1 in _chosen:
        sd[f"{_base}.weight"] = torch.zeros(_sh0, _sh1, dtype=torch.uint8)
    return len(_chosen), "(minimal placeholders)"


def register_aimdo_hooks(diffusion_model):
    pass


def _flush_tint4_caches(diffusion_model):
    from .tint4_loader import TINT4Linear
    for m in diffusion_model.modules():
        if isinstance(m, TINT4Linear):
            m.release_xpu()


def patch_model_for_aimdo(model):
    st = aimdo_state()
    if st == "none":
        return
    log.info("[TINT4 AIMDO] state=%s — detach wrapper active", st)

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
