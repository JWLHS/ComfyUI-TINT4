"""
tint4_lora_cache.py — TINT4 LoRA Disk Cache v1.0

Stores raw safetensors key → normalized path mappings (~30KB per LoRA).
Loading uses load_torch_file (bulk) + cache for O(1) dict access.

Cache file: {plugin_dir}/lora_cache/{sha256}.json
"""
import os
import json
import hashlib
import logging

log = logging.getLogger("TINT4-LoRA-Cache")

CACHE_VERSION = 2


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _cache_path(lora_path: str) -> str:
    lora_hash = _sha256_file(lora_path)
    if not lora_hash:
        return ""
    cache_dir = os.path.join(os.path.dirname(__file__), "lora_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{lora_hash}.json")


def _extract_role_suffix(key: str) -> str | None:
    lower = key.lower()
    if "lokr_w1" in lower:
        return "lokr_w1"
    if "lokr_w2" in lower:
        return "lokr_w2"
    if "lora_up" in lower or ".lora_b." in lower or lower.endswith(".lora_b"):
        return "up"
    if "lora_down" in lower or ".lora_a." in lower or lower.endswith(".lora_a"):
        return "down"
    if lower.endswith(".alpha"):
        return "alpha"
    return None


def _strip_role(key: str, role: str) -> str:
    variants = {
        "up": [".lora_up.", ".lora_up", ".lora_B.", ".lora_B"],
        "down": [".lora_down.", ".lora_down", ".lora_A.", ".lora_A"],
        "lokr_w1": [".lokr_w1.", ".lokr_w1"],
        "lokr_w2": [".lokr_w2.", ".lokr_w2"],
        "alpha": [".alpha"],
    }
    for v in variants.get(role, []):
        idx = key.lower().find(v.lower())
        if idx != -1:
            base = key[:idx].rstrip("._")
            if base.endswith(".weight"):
                base = base[:-len(".weight")]
            return base
    return key


def save_lora_cache(
    lora_path: str,
    format_type: str,
    lora_sd_keys: list,
) -> bool:
    cache_file = _cache_path(lora_path)
    if not cache_file:
        return False

    lora_hash = _sha256_file(lora_path)
    if not lora_hash:
        return False

    from .tint4_lora_common import _normalize_layer_path

    base_map: dict[str, dict] = {}
    for raw_key in lora_sd_keys:
        role = _extract_role_suffix(raw_key)
        if role is None:
            continue
        base = _strip_role(raw_key, role)
        if base not in base_map:
            base_map[base] = {}
        base_map[base][role] = raw_key
        if role == "alpha":
            alpha_base = raw_key.rsplit(".alpha", 1)[0] if raw_key.endswith(".alpha") else base
            base_map.setdefault(alpha_base, {})["__alpha_key__"] = raw_key

    entries = {}
    for base, roles in base_map.items():
        norm = _normalize_layer_path(base)
        if norm is None:
            continue
        if "lokr_w1" in roles or "lokr_w2" in roles:
            lora_type = "lokr"
        else:
            lora_type = "standard"

        entry = {
            "type": lora_type,
            "alpha": None,
            "is_qkv": norm.endswith(".attn.qkv") if norm else False,
        }
        for role, raw_key in roles.items():
            if role == "__alpha_key__":
                continue
            entry[f"raw_{role}"] = raw_key
        if "__alpha_key__" in roles:
            entry["has_alpha_key"] = roles["__alpha_key__"]
        else:
            entry["has_alpha_key"] = None
        if lora_type == "standard" and "up" in roles:
            entry["alpha_from_shape"] = True
        entries[norm] = entry

    payload = {
        "version": CACHE_VERSION,
        "lora_hash": lora_hash,
        "format": format_type,
        "entries": entries,
    }

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        size_kb = os.path.getsize(cache_file) / 1024
        log.info(
            f"[Cache] Saved → {os.path.basename(cache_file)} "
            f"({size_kb:.1f} KB, {len(entries)} layers)"
        )
        return True
    except Exception as e:
        log.warning(f"[Cache] Failed to save: {e}")
        return False


def load_lora_cache(lora_path: str) -> dict | None:
    cache_file = _cache_path(lora_path)
    if not cache_file or not os.path.exists(cache_file):
        return None

    lora_hash = _sha256_file(lora_path)
    if not lora_hash:
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != CACHE_VERSION:
        log.info("[Cache] Version mismatch, rebuilding")
        return None
    if payload.get("lora_hash") != lora_hash:
        log.info("[Cache] LoRA file changed, rebuilding")
        return None

    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        return None

    log.info(
        f"[Cache] Hit → {os.path.basename(lora_path)} "
        f"({len(entries)} layers)"
    )
    return {
        "format": payload.get("format", "standard"),
        "entries": entries,
    }


def clear_lora_cache(lora_path: str) -> bool:
    cache_file = _cache_path(lora_path)
    if cache_file and os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            log.info(f"[Cache] Deleted: {os.path.basename(cache_file)}")
            return True
        except OSError as e:
            log.warning(f"[Cache] Failed to delete: {e}")
    return False
