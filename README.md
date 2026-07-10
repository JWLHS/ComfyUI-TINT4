---
# TINT4 v1.0 — torchao INT4 量化推理 for ComfyUI

基于 [torchao](https://github.com/pytorch/ao) 的模型量化与推理插件。支持 Intel XPU / NVIDIA CUDA / AMD ROCm。

> **v1.0 重大更新**：LoRA 系统全面重构——索引表 O(1) 匹配、GPU pre-hook 延迟注入、轻量 JSON 缓存、IS_CHANGED 精确跳过。首次加载 < 5s，参数不变 0s，改强度 < 1s。

---

## 安装

```
  1. 安装插件
     ComfyUI Manager 搜 "TINT4" 安装
     或 git clone 到 custom_nodes/ComfyUI-TINT4-XPU

  2. 安装 torchao 0.17.0+xpu（仅 Intel XPU 用户）
     pip install torchao>=0.17.0
     NVIDIA/CUDA 用户: pip install torchao>=0.17.0
     AMD/ROCm 用户: pip install torchao>=0.17.0 --index-url https://download.pytorch.org/whl/rocm6.4
###  INTEL用户 : pip install torchao --index-url https://download.pytorch.org/whl/xpu
     Intel XPU users: after installing torchao, run fix_torchao_xpu.bat (included in the plugin) to patch missing submodules in the XPU fork.
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

  3. 运行修复脚本
     双击 custom_nodes/ComfyUI-TINT4-XPU/fix_torchao_xpu.bat

  4. 重启 ComfyUI
```

# 一些量化后的模型和示例工作流，持续添加...
### https://pan.quark.cn/s/a324b2c9881b

---

## 节点

| 节点 | 功能 |
|---|---|
| **TINT4 Model Quantizer** | 将 fp16/bf16/fp8/int8 模型量化为 INT4 并保存为 safetensors |
| **TINT4 Model Loader** | 加载量化模型 |
| **TINT4 LoRA Loader** | 单 LoRA 热加载（标准 + LoKr + QKV fused/independent 全兼容） |
| **TINT4 LoRA Stack** | 多 LoRA 栈（最多 5 个） |

---

## v1.0 LoRA 加载优化详解

v1.0 对 LoRA 加载链路进行了端到端重构。以下是每一项优化的具体效果：

### 1. 索引表 O(1) 模块匹配

| | v0.x | v1.0 |
|---|------|------|
| 匹配方式 | `named_modules()` 遍历全部 ~1000 个模块，逐层 `isinstance` 检查 | 模型加载时建 `_tint4_lora_index` 路径→模块字典 |
| 单次查找 | O(n)，~2s | O(1)，< 0.1ms |
| 可靠性 | 依赖路径规范化一致，QKV 层易失配 | 精确到目标模块引用，QKV 自动展开 |

### 2. IS_CHANGED 不重复执行

| | v0.x | v1.0 |
|---|------|------|
| 实现 | `return random.random()` | `return (lora_name, strength, id(model.model))` |
| 行为 | 每次 prompt 都重新加载 LoRA（含 IO + 解析 + 注入） | 参数不变时 ComfyUI 完全跳过节点 |
| 典型耗时 | 每轮 5~30s | **0s** |

### 3. Bake-in 计算：CPU → GPU pre-hook

| | v0.x | v1.0 |
|---|------|------|
| 计算时机 | 注入时 CPU 离线算 `B@A` 大矩阵 | 延迟至模块 forward 前，GPU 计算 |
| 瓶颈层（txtfusion MLP, 16384×16384） | 单层 **1.5~2.0s**（CPU 满秩矩阵乘） | 单层 **< 1ms**（Xe Matrix Engine） |
| YFG GPT LoRA 全量加载 | ~32s | **0.63s** |
| Onyx_V1 全量加载 | ~185s | **4.78s** |

### 4. 轻量 JSON 磁盘缓存

| | v0.x | v1.0 |
|---|------|------|
| 缓存 | 无，每次全量解析 LoRA keys | JSON ~30KB，存 key → 模块路径映射 |
| 存储位置 | — | `插件目录/lora_cache/{sha256}.json` |
| 首次加载 | 5~30s（IO + 字符串规范化 + 遍历匹配） | 同上 + 写缓存 |
| 再次加载 | 同上（每轮重复） | **< 1s**（缓存命中，免解析免匹配） |
| 缓存失效 | — | LoRA 文件被替换 → SHA256 变化 → 自动重生成 |

### 5. 实测性能对比（Krea2 Turbo, Arc A770 16GB, LOW_VRAM）

| 场景 | v0.x | v1.0 | 提升 |
|------|------|------|------|
| 首次加载 YFG GPT LoRA（256层） | ~10s（保守估计，旧版无计时） | **0.63s** | **15×+** |
| 同 LoRA 改强度（缓存命中） | ~10s | **0.49s** | **20×+** |
| 参数完全不变 | ~10s | **0s** | **∞** |
| 首次加载 Onyx_V1（264层） | ~185s | **4.78s** | **38×** |
| 小 LoRA（BreastSlider, 104层） | ~0.3s | **0.05s** | 6× |



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

| 操作 | 效果 | 耗时 |
|------|------|------|
| 将 `strength` 设为 `0.0` | 等效关闭该 LoRA | 即时生效 |
| 切换为其他 LoRA 文件 | 替换为新 LoRA | < 1s（缓存命中） |
| 调整 strength 数值 | 仅改变强度 | < 1s |

---

## 已知问题：LoRA 节点与中间节点交互

**症状**：当 LoRA Loader / Stack 与 Model Loader 或采样器之间存在**其他中间节点**（如采样算法节点、开关节点等），Bypass LoRA 节点后可能出现残留或无法重新启用。

**原因**：ComfyUI 的 Bypass 机制在跳过节点时不执行任何清理逻辑。中间节点可能缓存或改变模型引用链，导致 LoRA 状态无法被正常清除。

**影响范围**：仅在 LoRA 节点被 Bypass **且**节点链中存在中间节点时发生。正常使用（strength=0 关闭、直接换 LoRA 文件）不受影响。

**临时解决方案**（任选其一）：

| 方案 | 操作 |
|---|---|
| **A. 强度归零** | 将 `strength` 改为 `0.0`，等效关闭 LoRA，即时生效 ✅ |
| **B. 更换 LoRA** | 直接在下拉菜单切换为其他 LoRA 或 `None`，即时生效 ✅ |
| **C. 删除节点** | 移除 LoRA Loader/Stack 节点，重连模型线（实际可能残留 1~2 次，建议优先 A/B） |
| **D. 添加清理节点** | 在工作流中加入任意能触发模型重载或缓存释放的节点 ✅ |

> 该问题不具致命性，后续版本将尝试通过前端状态监控 + 自动清理机制修复。

---

## 支持模型

| 模型类型 | 状态 |
|---|---|
| `krea2` | ✅ 实测通过（LoRA 全功能正常） |
| `flux2` | ✅ 实测通过（含 QuaRot ON 模型） |
| `boogu` | ✅ 架构检测已修复 |
| `z-image` | ✅ LoRA 生效正常 |
| `wan` / `ltx2` / `qwen` / `ernie` / `hidream` / `chroma` / `ideogram4` | ⚠️ 排除列表已配置，等待社区反馈 |
| `auto` | 空白排除列表，按需使用 |

---

## 性能参考

| 模型 | 原始大小 | INT4 大小 | 说明 |
|---|---|---|---|
| Krea2 Turbo (28 层) | ~24 GB | ~6 GB | 推理速度接近原生，LoRA < 1s |
| Flux2 Klein 9B (42 层) | ~18 GB | ~4.5 GB | 含 QuaRot 旋转，LoRA 正常 |
| Z-Image Turbo (30 层) | ~11 GB | ~5 GB | dim=3840 大模型，RAM 占用较高但已比 FP16 少 ~30% |

> 实测环境: Intel Arc A770 16GB, torch 2.12.1+xpu, ComfyUI 0.27.0, LOW_VRAM 模式。
> 实际 VRAM / 速度因分辨率和采样步数而异。

---

## 贡献

---
鸣谢 deeoseek v4 pro 
鸣谢 torchao项目 https://github.com/pytorch/ao

欢迎提交 Issue / PR。新模型支持、bug 修复、文档改进均可。

## 许可证

MIT
```

---

