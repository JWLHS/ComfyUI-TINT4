
# TINT4 v1.1 — torchao INT4 Quantized Inference for ComfyUI

> [中文版](README_CN.md)

Built on [torchao](https://github.com/pytorch/ao).  Supports Intel XPU / NVIDIA CUDA / AMD ROCm.

> **v1.1**: Reliable IS_CHANGED, _lora_needs_reset flag, isolated AIMDO bridge, single‑LoRA chaining, Stack slots 5→8, 🐍 plugin integration.

---

# Pre‑quantized models & example workflows:
### https://pan.quark.cn/s/a324b2c9881b

---

## Installation

```
  1. Install the plugin
     ComfyUI Manager → search "TINT4"
     or git clone into custom_nodes/ComfyUI-TINT4

  2. Install torchao
     pip install torchao>=0.17.0
     Intel XPU:   pip install torchao --index-url https://download.pytorch.org/whl/xpu
     NVIDIA/CUDA: pip install torchao>=0.17.0
     AMD/ROCm:    pip install torchao>=0.17.0 --index-url https://download.pytorch.org/whl/rocm6.4

  3. Intel XPU users: after installing torchao, double‑click fix_torchao_xpu.bat
     (included in the plugin directory) to patch missing sub‑modules

  4. Restart ComfyUI
```

---

## torchao XPU Fork — Missing Sub‑modules Fix

Intel's `torchao 0.17.0+xpu` omits several standard torchao sub‑modules, causing diffusers / transformers to crash on startup (`ModuleNotFoundError`).

**Auto‑fix (recommended)**: double‑click `fix_torchao_xpu.bat` in the plugin directory, or run:

```bash
python fix_torchao_xpu.py
```

The script auto‑detects the torchao install path and creates all missing files.  Only affects `+xpu` builds; standard torchao is skipped.

**Manual fix**: if the script cannot run, create the following empty files under ComfyUI's `site-packages/torchao/`:

```
dtypes/floatx/__init__.py
dtypes/floatx/float8_layout.py      # content: Float8AQTTensorImpl = None
dtypes/uintx/__init__.py
dtypes/uintx/uint4_layout.py        # content: UInt4Tensor = None
dtypes/uintx/plain_layout.py
quantization/linear_quant.py
```

---

## Nodes

| Node | Purpose |
|---|---|
| **TINT4 Model Quantizer** | Quantize fp16/bf16/fp8/int8 → INT4 safetensors |
| **TINT4 Model Loader** | Load quantized model |
| **TINT4 LoRA Loader** | Single LoRA (standard + LoKr + QKV fused/independent) |
| **TINT4 LoRA Stack** | Multi‑LoRA (up to 8) |

---

## v1.1 Highlights

### 1. Reliable IS_CHANGED

v1.0's IS_CHANGED used `id(model.model)` to detect changes.  When ComfyUI cached the model object, the Python id stayed the same → the LoRA node was skipped → but `model.detach()` had already cleared LoRA state at the end of the previous run → **LoRA silently lost on the second Queue**.

v1.1 uses `random()` so the LoRA node **always executes**.  Cache‑hit reload only rebuilds CPU entries (< 1 s).  **Reliability over "0‑s skip" marketing.**

### 2. `_lora_needs_reset` Flag (inspired by [WINT4](https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta))

`model.detach()` sets the flag to `True` after clearing LoRA state.  LoRA Loader / Stack check it on entry:

- **True** → lingering or fresh‑loaded model → full clear → inject → set `False`
- **False** → previous node already handled cleanup → skip full clear → inject only

**This is the foundation for single‑LoRA chaining.**  When two Loaders are chained, the first clears globally and sets False; the second sees False and preserves the first's injection.

### 3. `_pop_module_lora` Enhancement

v1.0's `_pop_module_lora` only removed entries from the dict but **never rolled back deltas already baked into weight**.  Changing strength (e.g. 1.0 → 0.5) caused old and new deltas to accumulate.

v1.1 adds baked‑delta rollback: subtract old deltas from weight → clear entries → inject new values.  **Changing strength no longer accumulates.**

### 4. LoRA Stack Slot Expansion

Expanded from 5 slots to **8**.  `INPUT_TYPES`, `IS_CHANGED`, and `apply` all use `range(1, N)` updated accordingly.

### 5. Isolated AIMDO Bridge

All AIMDO logic is extracted into `tint4_aimdo.py`, decoupled from `tint4_loader.py`.  Future AIMDO upstream changes only require editing this file.

| Scenario | Placeholder strategy | RAM |
|----------|---------------------|-----|
| AIMDO ON | Full zero‑placeholders (224 layers) → VBAR skips fp16 allocation | ~30 GB |
| AIMDO OFF | 6‑placeholder → lightweight ModelPatcher, no fp16 CPU copies | ~15–20 GB |
| AIMDO ON + LoRA | No forward hooks → no repeated `_qt` rebuild | Normal speed |

> **Note**: `TINT4Linear` quantization data (`_qdata` / `_scale` / `_zp`) are plain Python attributes, invisible to `named_parameters()`.  VBAR cannot manage them.  `pin_weight` / `unpin_weight` are no‑ops on TINT4Linear (no `_v` attribute); v1.1 removes these ineffective calls.

---

## 🐍 Plugin Integration (ComfyUI-Custom-Scripts)

The 🐍 plugin provides LoRA info popups (CivitAI link, preview image, tags, training params, etc.).

### Single LoRA Loader

In ComfyUI settings → 🐍 Model Info → LoRA Nodes/Widgets, append:

```
,TINT4LoRALoader.lora_name
```

### LoRA Stack (two steps)

**Step 1**: Edit `ComfyUI-Custom-Scripts/web/js/modelInfo.js`.
Find:

```javascript
if (!value || value === "None") {
    return;            // ← change to continue
}
```

Change `return` to `continue`.  The original logic iterates the Stack's 8 slots but `return`s on the first `"None"` (empty slot), never checking subsequent filled slots.

**Step 2**: Add per‑slot entries in 🐍 settings:

```
,TINT4LoRAStack.lora_name_1,TINT4LoRAStack.lora_name_2,TINT4LoRAStack.lora_name_3,TINT4LoRAStack.lora_name_4,TINT4LoRAStack.lora_name_5,TINT4LoRAStack.lora_name_6,TINT4LoRAStack.lora_name_7,TINT4LoRAStack.lora_name_8
```

> If you change the slot count (e.g. to 10), update this list accordingly.

---

## v1.0 LoRA Optimizations (preserved)

v1.0's end‑to‑end LoRA pipeline overhaul is fully retained in v1.1.

### Index‑based O(1) Matching

| | v0.x | v1.0 / v1.1 |
|---|------|-------------|
| Matching | `named_modules()` traverse ~1000 modules | `_tint4_lora_index` dict built at load time |
| Lookup | O(n), ~2 s | O(1), < 0.1 ms |
| QKV | Partial format support | Full fused QKV + independent wq/wk/wv, auto‑slice |

### Bake‑in: CPU → GPU pre‑hook

| | v0.x | v1.0 / v1.1 |
|---|------|-------------|
| Compute | CPU offline `B@A` | GPU pre‑hook, lazy |
| Bottleneck layer (16384×16384) | 1.5–2.0 s | < 1 ms |
| Onyx_V1 full load | ~185 s | **~4.78 s** |

### Lightweight JSON Disk Cache

| | v0.x | v1.0 / v1.1 |
|---|------|-------------|
| Cache | None, full parse every run | JSON ~30 KB, SHA256‑indexed |
| First load | 5–30 s | Same + cache write |
| Subsequent loads | Same (every run) | **< 1 s** (cache hit, skip parse) |

### IS_CHANGED Evolution

| | v0.x | v1.0 | v1.1 |
|---|------|------|------|
| Implementation | `random()` | `(name, strength, id(model.model))` | `random()` |
| Behavior | Always executes | Skip on cache hit → **cross‑Queue failure** ❌ | Always executes ✅ |
| Reliability | ✅ (no cache) | ❌ (depends on external cache) | ✅ |

### Benchmarks (Krea2 Turbo, Arc A770 16 GB)

| Scenario | v0.x | v1.0 / v1.1 |
|----------|------|-------------|
| First LoRA load (no cache) | ~10 s | **0.6–5 s** |
| Same LoRA, new strength (cached) | ~10 s | **< 1 s** |
| Switch LoRA (cached) | ~10 s | **< 1 s** |

---

## Quantization Workflow

### Step 1: Analyze the source model

```bash
cd ComfyUI/custom_nodes/ComfyUI-TINT4
python analyse_quant.py <model.safetensors> <model_type>
```

The script outputs:
- Compatible group_size per layer (32/64/128/256)
- QuaRot recommendation (outlier ratio)
- Edge‑layer flags (should be excluded)

Example:
```
Recommend  GS compatible            Outlier  Type  Layer name
✅ Quant   [32, 64, 128, 256]      0.000    ATTN  blocks.0.attn.qkv.weight
⚠️ QuaRot  [64, 128]               0.210    ATTN  blocks.3.attn.qkv.weight
❌ Skip    [32]                     0.000    FFN   first.weight

🎯 Suggestion: group_size=128, QuaRot=ON
```

### Step 2: Quantize in ComfyUI

1. Add **TINT4 Model Quantizer** node
2. Select source model, model type, QuaRot / group_size / output filename
3. Queue

### Parameter Guide

| Parameter | Recommendation |
|---|---|
| `enable_quarot` | ON when outlier ratio > 0.15 |
| `group_size` | 128 (standard) / 64 (high‑quality) / 32 (Boogu forced) |
| `device` | xpu / cuda / cpu (auto‑detected) |

---

## Boogu Special Case

**Boogu‑Image (OmniGen2‑derived)** has layers with `in_features < 64`, incompatible with group_size=128.

The quantizer auto‑overrides to `group_size=32` (log: `[TINT4] Boogu: overriding group_size → 32`).  No manual adjustment needed.

---

## Analyse Script

`analyse_quant.py` analyzes raw FP16 models before quantization.

```bash
python analyse_quant.py model.safetensors model_type
```

Supports all model types: `krea2`, `flux2`, `z-image`, `wan`, `boogu`, `qwen`, `ernie`, `hidream`, `ltx2`, `chroma`, `ideogram4`, `anima`, `auto`.

---

## Using LoRAs

### LoRA Loader / Stack

1. Insert TINT4 LoRA Loader / Stack after TINT4 Model Loader
2. Select LoRA file and strength
3. For multiple LoRAs, use Stack (up to 8) or chain single Loaders

### Changing LoRA Effect

| Action | Effect | Overhead |
|--------|--------|----------|
| Set `strength` to `0.0` | Disables that LoRA | Instant |
| Switch to another LoRA file | Replace with new LoRA | < 1 s (cached) |
| Adjust strength value | Change intensity only | < 1 s |

---

## Known Issues

### Bypass + Intermediate Nodes

**Trigger**: when intermediate nodes (samplers, switches, etc.) sit between the Model Loader and a LoRA Loader/Stack that is Bypassed.

**Symptom**: LoRA effect persists after Bypass, or fails to re‑enable after un‑bypassing.

**Cause**: ComfyUI's Bypass mechanism runs no cleanup logic.  Intermediate nodes may cache or alter model reference chains.

**v1.1 mitigation**: the JS monitor (`tint4_monitor.js`) detects Bypass actions → sends a clear signal → Model Loader forces a LoRA reset on the next execution.

**Manual workarounds**:

| Option | Action |
|---|---|
| **A. Zero strength** | Set `strength` to `0.0` — instant ✅ |
| **B. Un‑bypass** | Restore the node → next Queue auto‑re‑injects ✅ |

---

## Supported Models

| Model | Status |
|---|---|
| `krea2` | ✅ Verified (LoRA fully functional) |
| `flux2` | ✅ Verified (incl. QuaRot ON models) |
| `boogu` | ✅ Architecture detection fixed |
| `z-image` | ✅ LoRA working |
| `wan` / `ltx2` / `qwen` / `ernie` / `hidream` / `chroma` / `ideogram4` / `anima` | ⚠️ Exclusion lists configured, awaiting community feedback |
| `auto` | Empty exclusion list |
| All legacy WINT4 models verified; TINT4 quantized models steadily progressing... | |

---

## Reference

| Model | Original | INT4 | Notes |
|---|---|---|---|
| Krea2 Turbo (28 layers) | ~24 GB | ~6 GB | Near‑native speed, LoRA < 1 s |
| Flux2 Klein 9B (42 layers) | ~18 GB | ~4.5 GB | QuaRot rotated, LoRA works |
| Z‑Image Turbo (30 layers) | ~11 GB | ~5 GB | dim=3840; RAM still ~30% less than fp16 |

> Tested on: Intel Arc A770 16 GB, torch 2.12.1+xpu, ComfyUI 0.27.0.
> Actual VRAM / speed varies with resolution and step count.

---

## Contributing

---

Thanks to deeoseek v4 pro
Thanks to the torchao project https://github.com/pytorch/ao

Issues / PRs welcome.  New model support, bug fixes, documentation improvements — all appreciated.

## License

MIT
```