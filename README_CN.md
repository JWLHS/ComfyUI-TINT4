
# TINT4 v1.1 — torchao INT4 量化推理 for ComfyUI

基于 [torchao](https://github.com/pytorch/ao) 的模型量化与推理插件。支持 Intel XPU / NVIDIA CUDA / AMD ROCm。

> **v1.1 更新**：IS_CHANGED 可靠执行、_lora_needs_reset 标志位、AIMDO 独立适配模块、单 LoRA 串联支持、Stack 槽位扩容至 8 个、🐍 插件接入方案。

---

# 一些量化后的模型和示例工作流，持续添加...
### https://pan.quark.cn/s/a324b2c9881b

---

## 安装

```
  1. 安装插件
     ComfyUI Manager 搜 "TINT4" 安装
     或 git clone 到 custom_nodes/ComfyUI-TINT4

  2. 安装 torchao
     pip install torchao>=0.17.0
     Intel XPU 用户: pip install torchao --index-url https://download.pytorch.org/whl/xpu
     NVIDIA/CUDA 用户: pip install torchao>=0.17.0
     AMD/ROCm 用户: pip install torchao>=0.17.0 --index-url https://download.pytorch.org/whl/rocm6.4

  3. Intel XPU 用户安装 torchao 后，双击插件目录下的 fix_torchao_xpu.bat 补全缺失子模块

  4. 重启 ComfyUI
```

---

## torchao XPU fork 缺少模块的解决方案

Intel 的 `torchao 0.17.0+xpu` 缺少标准 torchao 的部分子模块，导致 diffusers / transformers 在启动时崩溃（报错 `ModuleNotFoundError`）。

**自动修复**（推荐）：双击插件目录下的 `fix_torchao_xpu.bat`，或手动运行：

```bash
python fix_torchao_xpu.py
```

脚本会自动检测 torchao 安装路径并补全所有缺失文件。仅对 +xpu 版本生效，标准 torchao 自动跳过。

**手动修复**：如果脚本无法运行，在 ComfyUI 的 `site-packages/torchao/` 下创建以下空文件：

```
dtypes/floatx/__init__.py
dtypes/floatx/float8_layout.py      # 内容: Float8AQTTensorImpl = None
dtypes/uintx/__init__.py
dtypes/uintx/uint4_layout.py        # 内容: UInt4Tensor = None
dtypes/uintx/plain_layout.py
quantization/linear_quant.py
```

---

## 节点

| 节点 | 功能 |
|---|---|
| **TINT4 Model Quantizer** | 将 fp16/bf16/fp8/int8 模型量化为 INT4 并保存为 safetensors |
| **TINT4 Model Loader** | 加载量化模型 |
| **TINT4 LoRA Loader** | 单 LoRA 热加载（标准 + LoKr + QKV fused/independent 全兼容） |
| **TINT4 LoRA Stack** | 多 LoRA 栈（最多 8 个） |

---

## v1.1 新特性

### 1. IS_CHANGED 可靠执行

v1.0 的 IS_CHANGED 使用 `id(model.model)` 判断模型是否变更。当 ComfyUI 缓存模型对象时，Python id 不变 → ComfyUI 判定节点"未变更"→ 跳过执行 → 但上轮结束时 `model.detach()` 已清空 LoRA 状态 → **二次 Queue 后 LoRA 失效**。

v1.1 改为 `random()` 强制每次 Queue 都执行 LoRA 节点。缓存命中时仅重建 CPU entries，耗时 < 1s。**可靠性优先于"0s 跳过"的营销卖点。**

### 2. _lora_needs_reset 标志位（借鉴 WINT4 架构） (https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta)

`model.detach()` 清空 LoRA 后将标志位置为 `True`。LoRA Loader / Stack 入口检查该标志：

- **True** → 存在残留或模型刚重载 → 清全场 → 注入 → 置 `False`
- **False** → 上一节点已清理完毕 → 不清全场 → 仅注入当前 LoRA

**这是单 LoRA 节点串联的基础。** 两个单 LoRA Loader 串联时，第一个清全场后置 False，第二个见到 False 即保留前者的注入结果。

### 3. _pop_module_lora 增强

v1.0 的 `_pop_module_lora` 仅清除 entries 字典中的条目，**未回退已 baked 到 weight 上的 delta**。修改 LoRA strength 时（如 1.0 → 0.5），旧 delta 未被减去即注入新值 → 累加错误。

v1.1 补上了 baked delta 回退逻辑：先从 weight 减去旧 `_applied` 列表中的 delta，再清除条目，最后注入新值。**改 strength 不再叠加。**

### 4. LoRA Stack 槽位扩容

从 5 个槽位扩展为 **8 个**。涉及 `INPUT_TYPES`、`IS_CHANGED`、`apply` 三处 `range(1, N)` 的同步修改。

### 5. AIMDO 独立适配模块

AIMDO 相关逻辑全部提取至 `tint4_aimdo.py`，与 `tint4_loader.py` 解耦。AIMDO 上游更新时只需修改此文件，不影响核心加载逻辑。

| 场景 | 占位符策略 | 内存 |
|------|-----------|------|
| AIMDO ON | 全量零占位符（224 层）→ VBAR 跳过 fp16 分配 | ~30 GB |
| AIMDO OFF | 6 层占位符 → ModelPatcher 轻量，不保留 fp16 副本 | ~15–20 GB |
| AIMDO ON + LoRA | 不注册 forward hooks → 不反复重建 `_qt` | 速度正常 |

> **注意**：`TINT4Linear` 的量化权重（`_qdata`/`_scale`/`_zp`）是普通 Python 属性，不在 `named_parameters()` 中，VBAR 无法管理它们。`pin_weight`/`unpin_weight` 对 TINT4 是空操作（无 `_v` 属性），v1.1 已移除这些无效调用。

---

## 🐍 插件接入方案（ComfyUI-Custom-Scripts）

🐍 插件可为 LoRA 节点提供模型信息弹窗（CivitAI 链接、预览图、标签、训练参数等）。

### 单 LoRA Loader

在 ComfyUI 右上角设置 → 🐍 Model Info → LoRA Nodes/Widgets 配置中，末尾添加：

```
,TINT4LoRALoader.lora_name
```

### LoRA Stack（需两步）

**第一步**：修改 `ComfyUI-Custom-Scripts/web/js/modelInfo.js` 
找到：

```javascript
if (!value || value === "None") {
    return;            // ← 改为 continue
}
```

将 `return` 改为 `continue`。原逻辑在遍历 Stack 的 8 个槽位时，遇到第一个 `"None"`（空槽）即 `return` 退出整个函数，导致后续有 LoRA 的槽位永不被检查。

**第二步**：在 🐍 设置中添加逐槽位条目：

```
,TINT4LoRAStack.lora_name_1,TINT4LoRAStack.lora_name_2,TINT4LoRAStack.lora_name_3,TINT4LoRAStack.lora_name_4,TINT4LoRAStack.lora_name_5,TINT4LoRAStack.lora_name_6,TINT4LoRAStack.lora_name_7,TINT4LoRAStack.lora_name_8
```

> 若槽位数量有调整（如自行改为 10），需同步更新此配置。

---

## v1.0 LoRA 加载优化详解（保留）

v1.0 对 LoRA 加载链路进行了端到端重构。以下优化在 v1.1 中全部保留。

### 索引表 O(1) 模块匹配

| | v0.x | v1.0 / v1.1 |
|---|------|-------------|
| 匹配方式 | `named_modules()` 遍历全部 ~1000 个模块 | 模型加载时建 `_tint4_lora_index` 路径→模块字典 |
| 单次查找 | O(n)，~2s | O(1)，< 0.1ms |
| QKV 处理 | 仅部分格式 | 全面兼容 fused QKV + 独立 wq/wk/wv，自动切片 |

### Bake-in 计算：CPU → GPU pre-hook

| | v0.x | v1.0 / v1.1 |
|---|------|-------------|
| 计算时机 | CPU 离线算 `B@A` 大矩阵 | 延迟至 forward 前 GPU 计算 |
| 瓶颈层（16384×16384） | 单层 1.5~2.0s | 单层 < 1ms |
| Onyx_V1 全量加载 | ~185s | **~4.78s** |

### 轻量 JSON 磁盘缓存

| | v0.x | v1.0 / v1.1 |
|---|------|-------------|
| 缓存 | 无，每次全量解析 | JSON ~30KB，SHA256 索引 |
| 首次加载 | 5~30s | 同上 + 写缓存 |
| 再次加载 | 同上（每轮重复） | **< 1s**（缓存命中免解析） |

### IS_CHANGED 演进

| | v0.x | v1.0 | v1.1 |
|---|------|------|------|
| 实现 | `random()` | `(lora_name, strength, id(model.model))` | `random()` |
| 行为 | 每轮必执行 | 模型复用 → 跳过 → **跨 Queue 失效** ❌ | 每轮必执行 ✅ |
| 可靠性 | ✅ 但无缓存 | ❌ 依赖外部缓存 | ✅ |

### 实测性能（Krea2 Turbo, Arc A770 16GB）

| 场景 | v0.x | v1.0 / v1.1 |
|------|------|-------------|
| 首次加载 LoRA（无缓存） | ~10s | **0.6~5s** |
| 同 LoRA 改强度（缓存命中） | ~10s | **< 1s** |
| 换用其他 LoRA（缓存命中） | ~10s | **< 1s** |

---

## 量化方案

### 第一步：分析原始模型（确定量化参数）

使用内置分析脚本：

```bash
cd ComfyUI/custom_nodes/ComfyUI-TINT4
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

支持所有模型类型：`krea2`, `flux2`, `z-image`, `wan`, `boogu`, `qwen`, `ernie`, `hidream`, `ltx2`, `chroma`, `ideogram4`, `anima`, `auto`。

---

## LoRA 使用

### LoRA Loader / Stack

1. 在 TINT4 Model Loader 之后插入 TINT4 LoRA Loader / Stack
2. 选择 LoRA 文件和强度
3. 多 LoRA 叠加时使用 Stack 节点（最多 8 个），或串联多个单 LoRA Loader

### 更改 LoRA 效果

| 操作 | 效果 | 耗时 |
|------|------|------|
| 将 `strength` 设为 `0.0` | 等效关闭该 LoRA | 即时生效 |
| 切换为其他 LoRA 文件 | 替换为新 LoRA | < 1s（缓存命中） |
| 调整 strength 数值 | 仅改变强度 | < 1s |

---

## 已知问题

### Bypass + 中间节点交互

**触发条件**：LoRA Loader / Stack 与 Model Loader 之间**存在其他中间节点**（采样算法节点、开关节点等），且 LoRA 节点被 Bypass。

**症状**：Bypass 后 LoRA 残留，或取消 Bypass 后无法重新启用。

**原因**：ComfyUI Bypass 不执行清理逻辑，中间节点可能缓存或改变模型引用链。

**v1.1 缓解措施**：JS 监控（`tint4_monitor.js`）检测 Bypass 动作 → 发清除信号 → Model Loader 下次执行时强制清空 LoRA。

**手动方案**：

| 方案 | 操作 |
|---|---|
| **A. 强度归零** | `strength` 改为 `0.0`，即时生效 ✅ |
| **B. 取消 Bypass** | 恢复节点 → 下轮 Queue 自动重注 ✅ |

---

## 支持模型

| 模型类型 | 状态 |
|---|---|
| `krea2` | ✅ 实测通过（LoRA 全功能正常） |
| `flux2` | ✅ 实测通过（含 QuaRot ON 模型） |
| `boogu` | ✅ 架构检测已修复 |
| `z-image` | ✅ LoRA 生效正常 |
| `wan` / `ltx2` / `qwen` / `ernie` / `hidream` / `chroma` / `ideogram4` / `anima` | ⚠️ 排除列表已配置，等待社区反馈 |
| `auto` | 空白排除列表，按需使用 |
| 旧win4常用模型全部验证成功，时间原因tint4量化稳步验证推进中....  |

---

## 参考

| 模型 | 原始大小 | INT4 大小 | 说明 |
|---|---|---|---|
| Krea2 Turbo (28 层) | ~24 GB | ~6 GB | 推理速度接近原生，LoRA < 1s |
| Flux2 Klein 9B (42 层) | ~18 GB | ~4.5 GB | 含 QuaRot 旋转，LoRA 正常 |
| Z-Image Turbo (30 层) | ~11 GB | ~5 GB | dim=3840 大模型，RAM 占用较高但已比 FP16 少 ~30% |

> 实测环境: Intel Arc A770 16GB, torch 2.12.1+xpu, ComfyUI 0.27.0。
> 实际 VRAM / 速度因分辨率和采样步数而异。

---

## 贡献

---

鸣谢 deeoseek v4 pro
鸣谢 torchao 项目 https://github.com/pytorch/ao

欢迎提交 Issue / PR。新模型支持、bug 修复、文档改进均可。

## 许可证

MIT
```