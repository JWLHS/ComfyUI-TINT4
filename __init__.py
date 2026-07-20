"""
TINT4 v1.1 — torchao INT4 quantized model inference.

v1.1: +HTTP bypass signal endpoint (JS → Python bridge)
      +auto dependency installer via install.py
"""
import os as _os
import json as _json

WEB_DIRECTORY = "./js"


def _register_signal_endpoint():
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


def _ensure_torchao() -> bool:
    from .install import ensure_dependencies
    return ensure_dependencies(auto_install=True)


if _ensure_torchao():
    from .tint4_quantizer import NODE_CLASS_MAPPINGS as _QM, NODE_DISPLAY_NAME_MAPPINGS as _QD
    from .tint4_loader import NODE_CLASS_MAPPINGS as _LM, NODE_DISPLAY_NAME_MAPPINGS as _LD
    from .tint4_lora_loader import NODE_CLASS_MAPPINGS as _LRM, NODE_DISPLAY_NAME_MAPPINGS as _LRD
    from .tint4_lora_stack import NODE_CLASS_MAPPINGS as _LSM, NODE_DISPLAY_NAME_MAPPINGS as _LSD
    from .tint4_loader_ltx import NODE_CLASS_MAPPINGS as _LLM, NODE_DISPLAY_NAME_MAPPINGS as _LLD
    from .tint4_loader_wan import NODE_CLASS_MAPPINGS as _LWM, NODE_DISPLAY_NAME_MAPPINGS as _LWD
    NODE_CLASS_MAPPINGS = {**_QM, **_LM, **_LRM, **_LSM, **_LLM, **_LWM}
    NODE_DISPLAY_NAME_MAPPINGS = {**_QD, **_LD, **_LRD, **_LSD, **_LLD, **_LWD}
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
