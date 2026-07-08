# TINT4 — torchao INT4 量化推理 for ComfyUI

基于 [torchao](https://github.com/pytorch/ao) 的模型量化与推理插件。支持 Intel XPU / NVIDIA CUDA / AMD ROCm。

---

## 安装

```
┌─────────────────────────────────────────────────────────┐
│  1. 安装插件                                              │
│     ComfyUI Manager 搜 "TINT4" 安装                       │
│     或 git clone 到 custom_nodes/ComfyUI-TINT4-XPU       │
├─────────────────────────────────────────────────────────┤
│  2. 安装 torchao 0.17.0+xpu（仅 Intel XPU 用户）         │
│     从你的 Intel 分发渠道安装                              │
│     NVIDIA/CUDA 用户: pip install torchao>=0.17.0        │
│     AMD/ROCm 用户: pip install torchao>=0.17.0           │
│                   --index-url https://download.pytorch.org/whl/rocm6.4 │
├─────────────────────────────────────────────────────────┤
│  3. 运行修复脚本                                          │
│     双击 custom_nodes/ComfyUI-TINT4-XPU/                 │
│          fix_torchao_xpu.bat                             │
├─────────────────────────────────────────────────────────┤
│  4. 重启 ComfyUI                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 节点

| 节点 | 功能 |
|---|---|
| **TINT4 Model Quantizer** | 将 fp16/bf16/fp8/int8 模型量化为 INT4 并保存为 safetensors |
| **TINT4 Model Loader** | 加载量化模型 |
| **TINT4 LoRA Loader** | 单 LoRA 热加载（标准 + LoKr 格式） |
| **TINT4 LoRA Stack** | 多 LoRA 栈（最多 5 个） |

---

## torchao XPU fork 缺少模块的解决方案

Intel 的 `torchao 0.17.0+xpu` 缺少标准 torchao 的部分子模块，导致 diffusers / transformers 在启动时崩溃（报错 `ModuleNotFoundError`）。

**自动修复**（推荐）：双击插件目录下的 `fix_torchao_xpu.bat`，或手动运行：

```bash
python fix_torchao_xpu.py
```

脚本会自动检测 torchao 安装路径并补全所有缺失文件。仅对 +xpu 版本生效，标准 torchao 自动跳过。

**手动修复**：如果脚本无法运行,在 ComfyUI 的 `site-packages/torchao/` 下创建以下空文件：

```
dtypes/floatx/__init__.py
dtypes/floatx/float8_layout.py      # 内容: Float8AQTTensorImpl = None
dtypes/uintx/__init__.py
dtypes/uintx/uint4_layout.py        # 内容: UInt4Tensor = None
dtypes/uintx/plain_layout.py
quantization/linear_quant.py
```

---

## 量化方案

### 第一步：分析原始模型（确定量化参数）

使用内置分析脚本：

```bash
cd ComfyUI/custom_nodes/ComfyUI-TINT4-XPU
python analyse_quant.py <原始模型路径.safetensors> <模型类型>
```

脚本输出：
- 每层兼容的 group_size（32/64/128/256）
- 是否建议开启 QuaRot（Hadamard 旋转提升精度）
- 边缘层标记（是否该排除不量化）

示例输出：
```
推荐       GS兼容                   异常值   类型   层名
✅ 量化    [32, 64, 128, 256]      0.000   ATTN   blocks.0.attn.qkv.weight
⚠️ QuaRot 推荐     [64, 128]       0.210   ATTN   blocks.3.attn.qkv.weight
❌ 跳过（边缘层）   [32]           0.000   FFN    first.weight

🎯 建议: group_size=128, QuaRot=ON
```

### 第二步：量化

在 ComfyUI 中：
1. 添加 **TINT4 Model Quantizer** 节点
2. 选择原始模型、模型类型、参数（QuaRot / group_size / 输出文件名）
3. Queue

### 参数选择指南

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `enable_quarot` | 异常值 > 0.15 时 ON | Hadamard 旋转提升 INT4 精度 |
| `group_size` | 128（标准）/ 64（高精度）/ 32（Boogu 强制） | 越小精度越高，文件略大 |
| `device` | xpu / cuda / cpu | 自动检测可用设备 |

---

## Boogu 模型特殊要求

**Boogu-Image (OmniGen2 衍生)** 部分层的 `in_features < 64`，无法使用 group_size=128。

量化器会自动强制 `group_size=32`（日志显示 `[TINT4] Boogu: overriding group_size → 32`）。用户无需手动调整。

---

## 量化检测脚本

`analyse_quant.py` 用于**量化前**分析原始 FP16 模型，确定最佳量化参数。

```bash
python analyse_quant.py 模型路径.safetensors 模型类型
```

- 支持所有模型类型：`krea2`, `flux2`, `z-image`, `wan`, `boogu`, `qwen`, `ernie`, `hidream`, `ltx2`, `chroma`, `ideogram4`, `auto`
- 输出直接给出量化建议，复制到 Quantizer 节点参数即可

---

## LoRA 使用

### LoRA Loader / Stack

1. 在 TINT4 Model Loader 之后插入 TINT4 LoRA Loader / Stack
2. 选择 LoRA 文件和强度
3. 多 LoRA 叠加时使用 Stack 节点（最多 5 个）

### 更改 LoRA 效果

- 将 `strength` 设为 0：等效于关闭该 LoRA，效果即时生效
- 更换为其他 LoRA：直接选择即可，效果即时生效
- **不要**绕过 LoRA 节点直接连线（Krea2/Flux2 等模型正常工作，但 Z-Image 有已知问题见下方）

---

## Z-Image LoRA 已知问题

**症状**：Z-Image (Lumina2) 模型使用 LoRA 后，直接绕过 LoRA 节点（Bypass）不会清除 LoRA 效果。

**原因**：Z-Image 有大量非量化层（`noise_refiner`、`context_refiner` 等），LoRA delta 直接写入权重（bake-in），ComfyUI LOW_VRAM 模式下恢复原始权重失败。

**解决方案**（三选一）：

| 方案 | 操作 |
|---|---|
| **A. 权重归零** | 将 LoRA 的 `strength` 改为 `0.0`，等效关闭。效果即时生效 ✅ |
| **B. 更换 LoRA** | 直接在下拉菜单切换为其他 LoRA 或 `None`，效果即时生效 ✅ |
| **C. 重启 ComfyUI** | 如果需要彻底清除且不打算继续使用 LoRA |

**不影响其他模型**（Krea2、Flux2 等开关即时生效）。

---

## 支持模型

| 模型类型 | 状态 |
|---|---|
| `krea2` | ✅ 实测通过（LoRA 开关正常） |
| `flux2` | ✅ 实测通过（LoRA 开关正常） |
| `boogu` | ✅ 架构检测已修复 |
| `z-image` | ⚠️ LoRA 绕过不清理（见上方已知问题） |
| `wan` / `ltx2` / `qwen` / `ernie` / `hidream` / `chroma` / `ideogram4` | ⚠️ 排除列表已配置，等待社区反馈 |
| `auto` | 空白排除列表，按需使用 |

---

## 性能参考

| 模型 | 精度 | VRAM | 速度 |
|---|---|---|---|
| Krea2 (28 层) | INT4 gs=128 | ~25 GB | ~20 s |
| Z-Image (30 层, dim=3840) | INT4 gs=128 | ~40 GB | ~25 s |

---

## 贡献

欢迎提交 Issue / PR。新模型支持、bug 修复、文档改进均可。

## 许可证

MIT