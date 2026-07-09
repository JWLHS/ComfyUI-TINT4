"""
TINT4 v1.1 — torchao INT4 quantized model inference.

v1.1: +HTTP bypass signal endpoint (JS → Python bridge).
v1.0: +lightweight LoRA cache, +index-based O(1) matching,
      +IS_CHANGED fix, +JS bypass monitor, +QuaRot global marks.
"""
import sys
import os as _os
import json as _json

_MIN_TAO = (0, 17, 0)

WEB_DIRECTORY = "./js"


def _register_signal_endpoint():
    """Register POST /custom/TINT4/signal for JS → Python bypass signal."""
    try:
        from server import PromptServer
        from aiohttp import web

        @PromptServer.instance.routes.post("/custom/TINT4/signal")
        async def _tint4_signal_handler(request):
            try:
                payload = await request.json()
                from .tint4_lora_common import _write_clear_signal
                _write_clear_signal(payload)
                return web.json_response({"status": "ok"})
            except Exception:
                return web.json_response({"status": "error"}, status=400)
    except ImportError:
        pass


_register_signal_endpoint()


def _ensure_torchao():
    try:
        import torchao
    except ImportError:
        print("\n[TINT4] ERROR: torchao not found.\n"
              "  Intel XPU:  install torchao==0.17.0+xpu\n"
              "  NVIDIA/CUDA:  pip install torchao>=0.17.0\n"
              "  If you just installed torchao, run fix_torchao_xpu.bat first.\n")
        return False
    try:
        v = tuple(int(x) for x in torchao.__version__.split("+")[0].split(".")[:3])
        if v < _MIN_TAO:
            print(f"\n[TINT4] ERROR: torchao {torchao.__version__} too old.\n")
            return False
    except (ValueError, IndexError):
        pass
    return True


if _ensure_torchao():
    from .tint4_quantizer import NODE_CLASS_MAPPINGS as _QM, NODE_DISPLAY_NAME_MAPPINGS as _QD
    from .tint4_loader import NODE_CLASS_MAPPINGS as _LM, NODE_DISPLAY_NAME_MAPPINGS as _LD
    from .tint4_lora_loader import NODE_CLASS_MAPPINGS as _LRM, NODE_DISPLAY_NAME_MAPPINGS as _LRD
    from .tint4_lora_stack import NODE_CLASS_MAPPINGS as _LSM, NODE_DISPLAY_NAME_MAPPINGS as _LSD
    NODE_CLASS_MAPPINGS = {**_QM, **_LM, **_LRM, **_LSM}
    NODE_DISPLAY_NAME_MAPPINGS = {**_QD, **_LD, **_LRD, **_LSD}
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
