"""
install.py — TINT4 Dependency Installer v1.0

Single source of truth for torchao installation.
Called by:
  - ComfyUI Manager (after plugin install)
  - python install.py          (manual)
  - __init__.py                (auto on ComfyUI startup)
"""
import sys
import os
import subprocess


MIN_TAO = (0, 17, 0)
PIP_TIMEOUT = 300


# ═══════════════════════════════════════════════════════════════
# device detection
# ═══════════════════════════════════════════════════════════════

def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return "rocm"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ═══════════════════════════════════════════════════════════════
# version check
# ═══════════════════════════════════════════════════════════════

def _parse_version(v: str) -> tuple:
    return tuple(int(x) for x in v.split("+")[0].split(".")[:3])


def check_installed() -> str | None:
    try:
        import torchao
        v = _parse_version(torchao.__version__)
        if v >= MIN_TAO:
            return None
        return "old"
    except ImportError:
        return "missing"


# ═══════════════════════════════════════════════════════════════
# pip helpers
# ═══════════════════════════════════════════════════════════════

def _run(cmd: list, timeout: int = PIP_TIMEOUT) -> subprocess.CompletedProcess:
    print(f"[TINT4] {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _print_output(proc: subprocess.CompletedProcess):
    for line in proc.stdout.split("\n"):
        s = line.strip()
        if s:
            print(f"  {s}")
    if proc.returncode != 0:
        print("[TINT4] ✗ failed:")
        for line in proc.stderr.strip().split("\n")[-10:]:
            if line.strip():
                print(f"     {line}")


# ═══════════════════════════════════════════════════════════════
# XPU — hardcoded + --isolated, mirrors your manual success
# ═══════════════════════════════════════════════════════════════

def _install_xpu():
    print()
    print("=" * 58)
    print("  TINT4 — Auto-installing torchao for Intel XPU")
    print("=" * 58)
    print()

    proc = _run([
        sys.executable, "-m", "pip", "install", "--isolated", "torchao",
        "--index-url", "https://download.pytorch.org/whl/xpu",
    ])
    _print_output(proc)

    if proc.returncode != 0:
        print("\n  Manual: pip install torchao --index-url https://download.pytorch.org/whl/xpu")
        return False

    _fix_mps()
    print()
    print("  ✓ Done — restart ComfyUI.")
    print("=" * 58)
    print()
    return True


# ═══════════════════════════════════════════════════════════════
# CUDA / ROCm / CPU — generic, no --isolated
# ═══════════════════════════════════════════════════════════════

_OTHER_DEVICES = {
    "cuda": {"name": "NVIDIA CUDA", "url": ""},
    "rocm": {"name": "AMD ROCm",    "url": "https://download.pytorch.org/whl/rocm6.4"},
    "cpu":  {"name": "CPU",         "url": "https://download.pytorch.org/whl/cpu"},
}


def _install_generic(device: str):
    info = _OTHER_DEVICES.get(device, _OTHER_DEVICES["cpu"])

    print()
    print("=" * 58)
    print(f"  TINT4 — Auto-installing torchao for {info['name']}")
    print("=" * 58)
    print()

    cmd = [sys.executable, "-m", "pip", "install", "torchao"]
    if info["url"]:
        cmd += ["--index-url", info["url"]]

    proc = _run(cmd)
    _print_output(proc)

    if proc.returncode != 0:
        manual = "pip install torchao"
        if info["url"]:
            manual += f" --index-url {info['url']}"
        print(f"\n  Manual: {manual}")
        return False

    print()
    print("  ✓ Done — restart ComfyUI.")
    print("=" * 58)
    print()
    return True


# ═══════════════════════════════════════════════════════════════
# MPS fix
# ═══════════════════════════════════════════════════════════════

def _fix_mps():
    try:
        import torchao
    except ImportError:
        return
    d = os.path.join(os.path.dirname(torchao.__file__),
                     "experimental", "ops", "mps")
    if not os.path.isdir(d):
        return
    import shutil
    shutil.rmtree(d)
    print("[TINT4] ✓ MPS module removed")


# ═══════════════════════════════════════════════════════════════
# entry
# ═══════════════════════════════════════════════════════════════

def ensure_dependencies(auto_install: bool = True) -> bool:
    device = detect_device()
    print(f"[TINT4] device={device}")

    if check_installed() is None:
        if device == "xpu":
            _fix_mps()
        return True

    if os.environ.get("TINT4_NO_AUTO_INSTALL") == "1":
        print("[TINT4] TINT4_NO_AUTO_INSTALL=1 — skip")
        return False

    if not auto_install:
        return False

    if device == "xpu":
        return _install_xpu()
    else:
        return _install_generic(device)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["xpu", "cuda", "rocm", "cpu"])
    p.add_argument("--no-install", action="store_true")
    a = p.parse_args()

    if a.no_install:
        s = check_installed()
        sys.exit(0 if s is None else 1)

    if a.device == "xpu":
        ok = _install_xpu()
    elif a.device:
        ok = _install_generic(a.device)
    else:
        ok = ensure_dependencies(True)
    sys.exit(0 if ok else 1)
