# Audio-Motion Alignment (AMA) 模块创新点汇总

> 基于 VCT (Vision-Centric Transformer, CVPR 2025) 的改进工作
> 
> 作者：[Your Name]
> 日期：2024-12 (更新)

---

## 📊 实验结果快速汇总 (截至 2024-12-03)

### S4 数据集 (Single-source)
| Method | mIoU | F-score | 备注 |
|--------|------|---------|------|
| VCT Baseline (作者) | 86.20% | 93.40% | 论文报告 |
| VCT Baseline (复现) | 83.78% | 92.16% | 我们复现 |
| **VCT + AMA (RAFT, bs4)** | **85.18%** | **92.75%** | ⭐ **+1.40%** |

### MS3 数据集 (Multi-sources)
| Method | mIoU | F-score | 备注 |
|--------|------|---------|------|
| VCT Baseline (作者) | 66.84% | 82.33% | 作者权重 |
| VCT + AMA (bs4) | 65.02% | 78.02% | ❌ 过拟合 |
| VCT + AMA + BRM (v1, bs4) | 63.72% | 79.37% | ❌ 过拟合更严重 |
| VCT + AMA + BRM (v2, 训练中) | - | - | 🔄 优化中 |

### 关键发现
1. ✅ **S4**: AMA 有效，+1.40% mIoU (85.18% vs 83.78%)
2. ❌ **MS3**: 数据太少 (296 videos)，严重过拟合
3. ⚠️ **BRM**: 边界损失下降太快，加剧MS3过拟合
4. 💡 **Batch Size**: 80GB A100，S4/MS3最大bs=4，SS最大bs=2
5. 📊 **SS**: 复现baseline 49.60%，低于论文报告51.20% (-1.6%)

---

### SS 数据集 (Semantic Segmentation)
| Method | mIoU | F-score | 备注 |
|--------|------|---------|------|
| VCT Baseline (论文报告) | 51.20% | 55.50% | 作者报告 |
| VCT Baseline (复现 best) | 49.60% | 54.00% | 我们复现 |
| VCT Baseline (仓库权重) | 49.98% | 54.33% | 官方 model_best_ss3 |
| **VCT + AMA + BRM (训练中)** | - | - | 🔄 进行中 |

**数据集状态**: ✅ 完整数据集 (v1m + v1s + v2, 7948 train videos, 71 classes)

---

## 1. 研究动机 (Motivation)

### 1.1 现有方法的问题

VCT 使用 **PPQG (Prototype Prompted Query Generation)** 模块从视觉特征中生成 Object Queries。然而，该方法存在一个关键缺陷：

> **静态视觉显著性问题 (Static Visual Saliency Problem)**：
> PPQG 仅依赖静态图像特征进行 Query 生成，容易将**静态但视觉显著的背景物体**（如色彩鲜艳的装饰品、高对比度的物体）错误识别为发声物体。

### 1.2 简单光流的局限性

一个直观的解决方案是引入光流 (Optical Flow) 来关注动态区域。但简单光流存在噪声问题：

> **运动噪声问题 (Motion Noise Problem)**：
> 光流会捕捉所有运动区域，包括**"动但没声音"**的物体（如风吹树叶、路过的行人），这些噪声会干扰发声物体的定位。

### 1.3 我们的解决方案

**Audio-Motion Alignment (AMA)**：利用音频特征作为 Query，光流/运动特征作为 Key/Value，通过交叉注意力机制，只激活那些**"既在运动、又与声音相关"**的区域。

---

## 2. 方法概述 (Method Overview)

### 2.1 整体框架

```
┌─────────────────────────────────────────────────────────────────┐
│                        VCT + AMA Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌─────────────────────────┐    │
│  │  Video   │    │  Audio   │    │     Frame Difference    │    │
│  │  Frames  │    │Spectrogram│   │      (Motion Map)       │    │
│  └────┬─────┘    └────┬─────┘    └───────────┬─────────────┘    │
│       │               │                      │                   │
│       ▼               ▼                      ▼                   │
│  ┌──────────┐    ┌──────────┐    ┌─────────────────────────┐    │
│  │  Visual  │    │  Audio   │    │      Flow Encoder       │    │
│  │ Backbone │    │ Encoder  │    │   (Conv → BN → ReLU)    │    │
│  └────┬─────┘    └────┬─────┘    └───────────┬─────────────┘    │
│       │               │                      │                   │
│       │               │    ┌─────────────────┘                   │
│       │               │    │                                     │
│       │               ▼    ▼                                     │
│       │         ┌─────────────────┐                              │
│       │         │   AMA Module    │  ← [Ours: Innovation]        │
│       │         │ Q=Audio, K/V=Flow│                             │
│       │         │ Cross-Attention │                              │
│       │         └────────┬────────┘                              │
│       │                  │                                       │
│       │                  ▼                                       │
│       │         ┌─────────────────┐                              │
│       │         │ Motion Weight   │                              │
│       │         │ Map (filtered)  │                              │
│       │         └────────┬────────┘                              │
│       │                  │                                       │
│       ▼                  ▼                                       │
│  ┌────────────────────────────────┐                              │
│  │      PPQG (ProtoVCQ/VCQ)       │                              │
│  │  feat = feat * (1 + λ * map)   │  ← Motion-Guided Reweighting │
│  └────────────────────────────────┘                              │
│                    │                                             │
│                    ▼                                             │
│           ┌─────────────────┐                                    │
│           │  Transformer    │                                    │
│           │    Decoder      │                                    │
│           └─────────────────┘                                    │
│                    │                                             │
│                    ▼                                             │
│           ┌─────────────────┐                                    │
│           │  Segmentation   │                                    │
│           │     Masks       │                                    │
│           └─────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心创新点

| 创新点 | 描述 | 解决的问题 |
|--------|------|-----------|
| **AMA Module** | Audio-Motion Cross-Attention | 过滤"动但没声"的噪声 |
| **Motion-Guided Reweighting** | 特征加权 `x * (1 + λ * m)` | 增强动态发声区域 |
| **Learnable λ** | 可学习的运动权重因子 | 自适应调节运动影响 |

---

## 3. 技术细节 (Technical Details)

### 3.1 Audio-Motion Alignment (AMA) 模块

#### 3.1.1 模块定义

**文件位置**: `models/modeling/transformer_decoder/audio_motion_alignment.py`

```python
class AudioMotionAlignment(nn.Module):
    """
    Audio-Motion Alignment (AMA) Module.
    
    Input:
        audio_feat: (B, C_audio)  - 音频特征向量
        flow_map: (B, 1, H, W)    - 光流/运动图
        
    Output:
        motion_weight_map: (B, 1, H, W) - 音频激活的运动权重图 [0,1]
    """
```

#### 3.1.2 网络结构

```python
# Flow Encoder: 将运动图映射到嵌入空间
self.flow_encoder = nn.Sequential(
    nn.Conv2d(flow_channels, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, embed_dim, kernel_size=1),
    nn.BatchNorm2d(embed_dim),
)

# Audio Projection: 投影音频特征
self.audio_proj = nn.Sequential(
    nn.Linear(audio_dim, embed_dim),
    nn.LayerNorm(embed_dim),
)

# Cross-Attention: Q/K/V projections
self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # for Audio
self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # for Flow
self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # for Flow

# Learnable temperature for attention sharpness
self.temperature = nn.Parameter(torch.ones(1))
```

#### 3.1.3 前向传播

```python
def forward(self, audio_feat, flow_map):
    B, C_flow, H, W = flow_map.shape
    
    # 1. Encode flow features
    flow_feat = self.flow_encoder(flow_map)  # (B, embed_dim, H, W)
    flow_feat = flow_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, embed_dim)
    
    # 2. Project audio features as Query
    audio_query = self.audio_proj(audio_feat).unsqueeze(1)  # (B, 1, embed_dim)
    
    # 3. Multi-head Cross-Attention
    Q = self.q_proj(audio_query)  # (B, 1, embed_dim)
    K = self.k_proj(flow_feat)    # (B, H*W, embed_dim)
    V = self.v_proj(flow_feat)    # (B, H*W, embed_dim)
    
    # 4. Scaled dot-product attention with learnable temperature
    scale = (head_dim ** -0.5) * self.temperature
    attn_weights = softmax(Q @ K^T * scale)  # (B, num_heads, 1, H*W)
    
    # 5. Average attention weights across heads → spatial saliency map
    motion_weight_map = attn_weights.mean(dim=1).view(B, 1, H, W)
    
    # 6. Normalize to [0, 1]
    motion_weight_map = (map - min) / (max - min + ε)
    
    return motion_weight_map
```

#### 3.1.4 数学公式 (用于论文)

**Cross-Attention:**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k} \cdot \tau}\right)V
$$

其中：
- $Q = W_q \cdot \text{AudioProj}(f_a)$，$f_a$ 是音频特征
- $K = W_k \cdot \text{FlowEnc}(m)$，$m$ 是运动图
- $V = W_v \cdot \text{FlowEnc}(m)$
- $\tau$ 是可学习的温度参数

**Motion Weight Map:**
$$
M_{ama} = \text{Normalize}\left(\frac{1}{H}\sum_{h=1}^{H} A_h\right)
$$

其中 $A_h$ 是第 $h$ 个注意力头的权重，$H$ 是头数。

---

### 3.2 Motion-Guided Query Reweighting

#### 3.2.1 修改位置

**文件**: `models/modeling/transformer_decoder/vision_centric_transformer_decoder.py`

**类**: `ProtoVCQ` 和 `VCQ`

#### 3.2.2 核心代码

```python
class ProtoVCQ(nn.Module):
    def __init__(self, ...):
        # ... existing code ...
        
        # [Ours: Motion-Guided] Learnable motion weighting factor λ
        self.motion_lambda = nn.Parameter(torch.ones(1))

    def forward(self, x, motion_map=None):
        x = self.pre_proj(x)
        
        # [Ours: Motion-Guided] Apply motion-guided reweighting
        if motion_map is not None:
            if motion_map.shape[-2:] != x.shape[-2:]:
                motion_map = F.interpolate(motion_map, size=x.shape[-2:], 
                                           mode='bilinear', align_corners=False)
            # Reweighting: x' = x * (1 + λ * m)
            x = x * (1.0 + self.motion_lambda * motion_map)
        
        # ... rest of query generation ...
```

#### 3.2.3 数学公式

**Feature Reweighting:**
$$
\hat{F}_v = F_v \odot (1 + \lambda \cdot M_{ama})
$$

其中：
- $F_v \in \mathbb{R}^{B \times C \times H \times W}$ 是视觉特征
- $M_{ama} \in \mathbb{R}^{B \times 1 \times H \times W}$ 是 AMA 输出的运动权重图
- $\lambda$ 是可学习的标量参数
- $\odot$ 表示逐元素乘法（广播）

---

### 3.3 数据流集成

#### 3.3.1 Motion Map 计算 (Dataset Mapper)

**文件**: `models/data/dataset_mappers/avss4_semantic_dataset_mapper.py`

```python
# Frame-to-frame motion maps (simplified optical flow)
motion_maps = []
for num_img in range(len(images)):
    if num_img == 0:
        motion_map = torch.zeros((1, H, W), dtype=torch.float32)
    else:
        # L1 difference between consecutive frames
        diff = torch.abs(images[num_img].float() - images[num_img-1].float())
        motion_map = diff.mean(dim=0, keepdim=True)  # Average across RGB
        motion_map = motion_map / motion_map.max()   # Normalize to [0, 1]
    motion_maps.append(motion_map)

dataset_dict["motion_maps"] = torch.stack(motion_maps, dim=0)
```

#### 3.3.2 AMA 调用 (Decoder)

**文件**: `models/modeling/transformer_decoder/vision_centric_transformer_decoder.py`

```python
def forward(self, x, audio_features, mask_features, mask=None, motion_maps=None):
    # ...
    
    if motion_maps is not None and len(motion_maps) > 0:
        raw_motion_map = torch.stack(motion_maps, dim=0)  # [bt, 1, H, W]
        
        # Global average pooling for audio features
        audio_feat_for_ama = audio_features.mean(dim=[2, 3])  # [B, 256]
        
        # Downsample motion map for efficiency
        motion_map_small = F.interpolate(raw_motion_map, size=(24, 24), 
                                         mode='bilinear', align_corners=False)
        
        # [Ours: AMA Module] Apply Audio-Motion Alignment
        motion_weight_map = self.ama_module(audio_feat_for_ama, motion_map_small)
        
        # Upsample to mask_features resolution
        motion_weight_map = F.interpolate(motion_weight_map, size=(h_m, w_m), 
                                          mode='bilinear', align_corners=False)
    
    # Pass to PPQG
    visual_querys, prototypes = self.visual_query_block(mask_features, motion_weight_map)
```

---

## 4. 修改文件清单

| 文件路径 | 修改类型 | 描述 |
|----------|----------|------|
| `models/modeling/transformer_decoder/audio_motion_alignment.py` | **新建** | AMA 模块定义 |
| `models/modeling/transformer_decoder/vision_centric_transformer_decoder.py` | 修改 | 导入 AMA，添加 `self.ama_module`，修改 `forward()` |
| `models/data/dataset_mappers/avss4_semantic_dataset_mapper.py` | 修改 | 添加 `motion_maps` 计算 |
| `models/data/dataset_mappers/avsms3_semantic_dataset_mapper.py` | 修改 | 添加 `motion_maps` 计算 |
| `models/vct_model.py` | 修改 | 传递 `motion_maps` 到 `sem_seg_head` |
| `models/modeling/meta_arch/vct_model_head.py` | 修改 | 传递 `motion_maps` 参数 |

---

## 5. 实验设置

### 5.1 数据集

- **S4 (Single-source)**: 4932 videos, 5 frames each
- **MS3 (Multi-sources)**: 424 videos, 5 frames each
- **AVSS (Semantic-labels)**: Full semantic segmentation

### 5.2 训练配置

```yaml
SOLVER:
  IMS_PER_BATCH: 4      # 默认 batch size (8 会 OOM)
  BASE_LR: 0.00014      # 对应 batch size 4 的学习率
  MAX_ITER: 20000
  
MODEL:
  MASK_FORMER:
    HIDDEN_DIM: 256
    NUM_OBJECT_QUERIES: 100
```

### 5.3 学习率缩放规则 (AdamW 优化器)

对于 **AdamW 优化器**，学习率与 batch size 关系使用 **平方根缩放** (Square Root Scaling)：

$$
lr_{new} = lr_{base} \times \sqrt{\frac{bs_{new}}{bs_{base}}}
$$

| Batch Size | 学习率 (BASE_LR) | 计算公式 |
|------------|------------------|----------|
| 2 (基准) | 0.0001 | 基准值 |
| 4 | 0.00014 | 0.0001 × √2 ≈ 0.00014 |
| **8** | **0.0002** | 0.0001 × √4 = **0.0002** |
| 16 | 0.00028 | 0.0001 × √8 ≈ 0.00028 |

> **注意**: Batch size 8 在 80GB A100 上会 OOM，实际最大可用 batch size 为 4。

### 5.3 评估指标

- **mIoU**: Mean Intersection over Union
- **F-score**: F-measure for binary segmentation

---

## 6. 实验结果

### 6.1 S4 数据集结果

| Method | mIoU | F-score | 相对提升 | 备注 |
|--------|------|---------|----------|------|
| VCT (论文报告) | 86.20% | 93.40% | - | 作者报告 |
| VCT (复现 baseline) | 83.78% | 92.16% | baseline | 我们复现 |
| **VCT + AMA v1 (帧差)** | **84.26%** | **92.40%** | **+0.48%** | Exp-001 |
| **VCT + AMA v2 (RAFT, bs2)** | **84.65%** | **92.39%** | **+0.87%** | Exp-002 |
| **VCT + AMA v3 (RAFT, bs4)** | **85.18%** | **92.75%** | **+1.40%** | Exp-004 ⭐ |

### 6.2 实验详情记录

#### Exp-001: AMA + 帧差法 (Frame Difference)

**实验日期**: 2024-11-28

**配置**:
```yaml
# 运动特征: 帧差法 (Frame Difference)
Motion Type: Frame L1 Difference
Motion Channels: 1
AMA embed_dim: 256
AMA num_heads: 4
PPQG λ: Learnable (init=1.0)

# 训练配置
SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.0001
  MAX_ITER: 45000  # S4 完整训练
  
# 数据
Dataset: S4 (4932 videos)
Input Size: 384x384
```

**训练日志**: `output/s4_ama_train.log`

**测试结果**:
```
mIoU: 0.8426 (+0.48% vs baseline)
F-score: 0.9240 (+0.24% vs baseline)
```

**分析**:
- ✅ 正向提升，证明 AMA 思路有效
- ⚠️ 提升幅度有限，可能原因:
  1. 帧差法噪声大（光照变化、相机抖动）
  2. 帧差只捕获 temporal gradient，缺乏 spatial motion structure
  3. AMA 模块容量较小

**下一步**: 使用 RAFT 预计算高质量光流

---

#### Exp-002: AMA + RAFT 光流 (RAFT Optical Flow)

**实验日期**: 2024-11-29

**配置**:
```yaml
# 运动特征: RAFT 光流 (预计算)
Motion Type: RAFT Optical Flow (magnitude normalized)
Motion Channels: 1
RAFT Model: raft_large (torchvision built-in)
AMA embed_dim: 256
AMA num_heads: 4
PPQG λ: Learnable (init=1.0)

# 训练配置
SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.0001
  MAX_ITER: 45000

# 数据
Dataset: S4 (4932 videos)
Input Size: 384x384
Flow Storage: /media/a100/.../AVS_dataset/raft_flow/s4/
```

**训练日志**: `output/s4_raft_ama_train.log`

**测试结果**:
```
mIoU: 0.8465 (+0.87% vs baseline, +0.39% vs 帧差)
F-score: 0.9239 (+0.23% vs baseline, -0.01% vs 帧差)
```

**分析**:
- ✅ RAFT 相比帧差有微小提升 (+0.39% mIoU)
- ⚠️ 但提升幅度远低于预期（预期 +1.5%）
- ⚠️ F-score 甚至略有下降

---

#### Exp-004: AMA + RAFT (Batch Size 4, LR 0.00014)

**实验日期**: 2024-12-01

**实验目的**: 增大 batch size 提升训练稳定性和最终精度

**配置**:
```yaml
# 运动特征: RAFT 光流 (预计算)
Motion Type: RAFT Optical Flow (magnitude normalized)
Motion Channels: 1
AMA embed_dim: 256
AMA num_heads: 4
PPQG λ: Learnable (init=1.0)

# 训练配置 (调整 batch size)
SOLVER:
  IMS_PER_BATCH: 4  # 增大 batch size (原 2)
  BASE_LR: 0.00014  # sqrt(2) 缩放 (原 0.0001)
  MAX_ITER: 45000

# 数据
Dataset: S4 (4932 videos)
Input Size: 384x384
```

**训练日志**: `output/s4_bs8_ama_train.log`

**测试结果**:
```
mIoU: 0.8518 (+1.40% vs baseline 0.8378)
F-score: 0.9275 (+0.59% vs baseline 0.9216)
```

**分析**:
- ✅ 相比 batch size 2 有显著提升 (+0.53% mIoU)
- ✅ 总体相比 baseline 提升 +1.40% mIoU
- ⚠️ 仍低于作者报告的 86.20%，但差距在缩小
- 💡 batch size 影响较大，说明之前可能训练不充分

---

### 6.2 实验结果深度分析

#### 6.2.1 为什么 AMA 提升有限？

| 可能原因 | 详细分析 | 验证方法 |
|----------|----------|----------|
| **S4 数据集特性** | S4 是单发声源场景，发声物体通常已经是画面中最显著的视觉目标，运动信息的边际收益有限 | 在 MS3 多源场景测试 |
| **运动与声音弱相关** | 并非所有发声物体都有明显运动（如静置的扬声器、正脸说话的人） | 可视化 motion weight map |
| **AMA 模块容量** | 简单的 Cross-Attention 可能不足以建模复杂的音视频关系 | 增加模块复杂度 |
| **光流质量** | RAFT 是通用光流，可能对 AVS 特定场景不够适配 | 分析失败案例 |
| **训练不充分** | AMA 模块参数较少，可能过早收敛 | 调整学习率/训练轮次 |

#### 6.2.2 Baseline 本身精度问题

我们复现的 baseline (83.78%) 比作者报告 (86.20%) 低约 **2.4%**。这可能导致:
- 我们的"提升"实际上只是弥补了 baseline 的不足
- 如果 baseline 正常，AMA 可能根本没有提升

**建议**: 先排查 baseline 为什么低于作者报告

#### 6.2.3 单发声源 vs 多发声源

| 任务 | 特点 | AMA 预期收益 |
|------|------|--------------|
| **S4 (单发声源)** | 只有一个目标，通常最显著 | 低 |
| **MS3 (多发声源)** | 多个目标，需要区分谁在发声 | **高** |
| **AVSS (语义)** | 更复杂的场景 | 中-高 |

**关键洞察**: AMA 的设计初衷是**过滤"动但没声"的噪声**。在 S4 中，这种噪声可能本来就很少！

---

#### Exp-003: MS3 + AMA + RAFT (多发声源场景验证)

**实验日期**: 2024-12-01 ~ 2024-12-02

**实验目的**: 验证 AMA 在多发声源场景 (MS3) 中的效果。理论上，MS3 更需要区分"谁在发声"，AMA 应该更有价值。

**配置**:
```yaml
# 运动特征: RAFT 光流 (预计算)
Motion Type: RAFT Optical Flow (magnitude normalized)
Motion Channels: 1
RAFT Model: raft_large (torchvision built-in)
AMA embed_dim: 256
AMA num_heads: 4
PPQG λ: Learnable (init=1.0)

# 训练配置 (从 S4+AMA bs4 权重初始化)
Initial Weights: S4 + AMA (bs4) model_best.pth
SOLVER:
  IMS_PER_BATCH: 4
  BASE_LR: 0.00014  # sqrt(2) 缩放
  MAX_ITER: 40000

# 数据
Dataset: MS3 (424 videos: 296 train, 64 val, 64 test)
Input Size: 384x384
Flow Storage: /media/a100/.../AVS_dataset/raft_flow/ms3/
```

**训练日志**: `output/ms3_bs4_ama_train.log`

**训练过程 mIoU 变化 (验证集)**:
```
iter 500:   0.6113 → iter 2500: 0.6461 → iter 16500: 0.6708 (峰值 ⭐)
→ iter 20000: 0.6569 → iter 30000: 0.6440 → iter 40000: 0.6411
```

**测试结果**:
```
验证集 Best: mIoU 0.6708, F-score 0.8354 (iter ~16500)
测试集 Final: mIoU 0.6502, F-score 0.7802 ❌
```

**MS3 Baseline (作者原始权重测试)**:
```
mIoU: 0.6684, F-score 0.8233
```

**问题分析**:
| 数据集 | AMA (model_best) | Baseline (无AMA) | 差距 |
|--------|------------------|------------------|------|
| 验证集 | **0.6708** ✅ | - | +0.24% |
| 测试集 | 0.6502 ❌ | **0.6684** | **-1.82%** |

**关键发现**:
1. ✅ AMA 在**验证集**上超过了 baseline (0.6708 > 0.6684)
2. ❌ 但在**测试集**上表现下降 (0.6502 < 0.6684)
3. 🔴 **严重的过拟合问题**: 模型过拟合到验证集的模式，泛化到测试集失败

**可能原因**:
1. **数据量太少**: MS3 只有 296 个训练视频，AMA 模块容易过拟合
2. **学习率太高**: 0.00014 可能对小数据集过大
3. **正则化不足**: AMA 模块缺少 dropout 或 weight decay
4. **训练过长**: 40000 iter 对 MS3 来说可能太长
5. **验证集-测试集分布不一致**: 导致验证集上的最优不是测试集上的最优

---

## 7. 论文写作建议

### 7.1 标题建议

- "Audio-Motion Alignment for Robust Audio-Visual Segmentation"
- "Learning to Align Audio and Motion for Sound Source Segmentation"
- "AMA: Filtering Motion Noise with Audio Guidance in AVS"

### 7.2 贡献点 (Contributions)

1. **问题发现**: 指出现有 AVS 方法存在的静态显著性误检和运动噪声问题
2. **方法创新**: 提出 Audio-Motion Alignment (AMA) 模块，通过音频-运动交叉注意力过滤噪声
3. **即插即用**: AMA 模块可以无缝集成到现有 AVS 框架中
4. **实验验证**: 在 AVSBench 数据集上验证方法有效性

### 7.3 Related Work 关键词

- Audio-Visual Learning
- Sound Source Localization
- Video Object Segmentation
- Cross-modal Attention
- Optical Flow in Video Understanding

---

## 8. 训练命令

### S4 数据集训练

```bash
cd /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS

# 前台运行（可以看到实时输出）
bash scripts/s4_swinb_384_train.sh

# 后台运行（断开 SSH 也能继续）
nohup bash scripts/s4_swinb_384_train.sh > output/s4_ama_train.log 2>&1 &

# 查看训练日志
tail -f output/s4_ama_train.log
```

### S4 数据集测试

```bash
bash scripts/s4_swinb_384_test.sh
```

### MS3 数据集训练

```bash
nohup bash scripts/ms3_swinb_384_train.sh > output/ms3_ama_train.log 2>&1 &
```

---

## 9. 代码注释规范

所有修改处都使用以下注释标记：

```python
# [Ours: AMA Module] 描述
# [Ours: Motion-Guided] 描述
```

便于后续代码审查和论文写作时快速定位修改位置。

---

## 10. 后续改进方向

### 10.1 已完成
- [x] ~~**真实光流**: 使用 RAFT 等预训练光流模型替代帧差~~ → 效果提升有限

### 10.2 高优先级 (推荐立即尝试)

#### 方案 A: 转战 MS3 数据集 ⭐⭐⭐⭐⭐
```
理由: 
- MS3 是多发声源场景，更需要区分"谁在发声"
- AMA 的核心价值 (过滤运动噪声) 在 MS3 中更能体现
- S4 提升有限可能是数据集特性决定的

操作:
1. 准备 MS3 数据集 (已有)
2. 计算 MS3 的 RAFT 光流
3. 训练 VCT + AMA on MS3
4. 对比 baseline
```

#### 方案 B: 可视化分析 + 失败案例 ⭐⭐⭐⭐
```
理由:
- 不盲目改进，先理解现有模块的行为
- 找到 AMA 没起作用的具体原因

操作:
1. 可视化 motion_weight_map (AMA 输出)
2. 对比 AMA vs 无 AMA 的预测差异
3. 分析: 哪些案例 AMA 帮助了？哪些没帮助？
```

#### 方案 C: 修正 Baseline ⭐⭐⭐⭐
```
理由:
- 我们 baseline 83.78% 比作者 86.20% 低 2.4%
- 可能有配置/数据处理问题
- 在正确的 baseline 上测 AMA 更有意义

操作:
1. 仔细对比作者配置
2. 检查数据预处理是否一致
3. 尝试不同随机种子
```

### 10.3 中优先级

4. **Audio-Motion Consistency Loss**: 添加辅助损失约束音频-运动对齐
5. **多尺度 AMA**: 在 Pixel Decoder 的多个尺度应用 AMA
6. **时序建模**: 使用 LSTM/Transformer 建模时序运动信息
7. **AMA 架构改进**: 尝试更复杂的融合策略 (如 Gated Fusion、Cross-Modal Transformer)

### 10.4 实验路线图 (建议)

```
Week 1: [当前] S4 + AMA (帧差/RAFT) → 效果有限 ✓
        
Week 2: MS3 + AMA → 验证多源场景收益
        可视化分析 → 理解 AMA 行为
        
Week 3: 根据 Week 2 结果选择方向:
        - 如果 MS3 提升大 → 完善方法，准备投稿
        - 如果 MS3 也不行 → 换创新点或深入分析原因
```

---

## 11. RAFT 光流预计算方案 (详细)

### 11.1 为什么需要 RAFT？

| 对比项 | 帧差法 (Frame Diff) | RAFT 光流 |
|--------|---------------------|-----------|
| **精度** | 低，只有 temporal gradient | 高，SOTA 光流算法 |
| **噪声** | 对光照、相机抖动敏感 | 鲁棒，经过大规模训练 |
| **语义** | 无，纯像素级差异 | 有，捕获真实运动结构 |
| **计算量** | 几乎为0 | 高，但可预计算 |
| **输出维度** | 1 通道 (magnitude) | 2 通道 (u, v) 或 magnitude |

### 11.2 RAFT 简介

**RAFT (Recurrent All-Pairs Field Transforms)** - ECCV 2020 Best Paper

- **论文**: "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
- **代码**: https://github.com/princeton-vl/RAFT
- **特点**:
  - 在 Sintel、KITTI 等光流数据集上 SOTA
  - 迭代式 refinement，精度高
  - 有多种预训练模型可选

### 11.3 实施方案

#### Step 1: 安装 RAFT

```bash
# 克隆 RAFT 仓库
cd /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT

# 下载预训练模型
./download_models.sh
# 或手动下载: https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT

# 预训练模型说明:
# - raft-things.pth: 在 FlyingThings3D 上训练，泛化性好
# - raft-sintel.pth: 在 Sintel 上微调，适合真实场景
# - raft-kitti.pth: 在 KITTI 上微调，适合驾驶场景
# 推荐使用 raft-things.pth 或 raft-sintel.pth
```

#### Step 2: 预计算光流脚本

创建 `avs_tools/compute_raft_flow.py`:

```python
"""
[Ours: RAFT Flow] 预计算 RAFT 光流
Usage:
    python avs_tools/compute_raft_flow.py \
        --dataset s4 \
        --input_root /path/to/AVS_dataset \
        --output_root /path/to/AVS_dataset/raft_flow \
        --model /path/to/RAFT/models/raft-sintel.pth
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加 RAFT 路径
sys.path.append('/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/RAFT/core')
from raft import RAFT
from utils.utils import InputPadder


def load_image(path):
    """加载图像并转换为 tensor"""
    img = np.array(Image.open(path)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()


def compute_flow(model, image1, image2):
    """计算光流"""
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    
    with torch.no_grad():
        _, flow = model(image1, image2, iters=20, test_mode=True)
    
    flow = padder.unpad(flow)
    return flow[0].cpu().numpy()  # [2, H, W]


def flow_to_magnitude(flow):
    """将光流转换为 magnitude (单通道)"""
    u, v = flow[0], flow[1]
    magnitude = np.sqrt(u**2 + v**2)
    # 归一化到 [0, 1]
    magnitude = magnitude / (magnitude.max() + 1e-6)
    return magnitude.astype(np.float32)


def main(args):
    # 加载 RAFT 模型
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module.cuda().eval()
    
    # 数据集路径配置
    if args.dataset == 's4':
        splits = ['train', 'val', 'test']
        frame_dir = 'visual_frames_384'
    elif args.dataset == 'ms3':
        splits = ['train', 'val', 'test']
        frame_dir = 'visual_frames_384'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    for split in splits:
        input_path = os.path.join(args.input_root, f'{args.dataset}_data_384', frame_dir, split)
        output_path = os.path.join(args.output_root, args.dataset, split)
        os.makedirs(output_path, exist_ok=True)
        
        videos = sorted(os.listdir(input_path))
        print(f"Processing {split}: {len(videos)} videos")
        
        for video in tqdm(videos, desc=f'{args.dataset}/{split}'):
            video_input = os.path.join(input_path, video)
            video_output = os.path.join(output_path, video)
            os.makedirs(video_output, exist_ok=True)
            
            frames = sorted([f for f in os.listdir(video_input) if f.endswith('.png')])
            
            for i in range(len(frames)):
                output_file = os.path.join(video_output, f'flow_{i:04d}.npy')
                
                if os.path.exists(output_file):
                    continue  # 跳过已计算的
                
                if i == 0:
                    # 第一帧：使用零光流
                    img = Image.open(os.path.join(video_input, frames[0]))
                    h, w = img.size[1], img.size[0]
                    flow_mag = np.zeros((h, w), dtype=np.float32)
                else:
                    # 计算 frame[i-1] -> frame[i] 的光流
                    img1 = load_image(os.path.join(video_input, frames[i-1]))
                    img2 = load_image(os.path.join(video_input, frames[i]))
                    flow = compute_flow(model, img1, img2)
                    flow_mag = flow_to_magnitude(flow)
                
                np.save(output_file, flow_mag)
    
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['s4', 'ms3', 'avss'])
    parser.add_argument('--input_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    
    # RAFT 模型参数 (保持默认)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    
    args = parser.parse_args()
    main(args)
```

#### Step 3: 运行预计算

```bash
# S4 数据集
python avs_tools/compute_raft_flow.py \
    --dataset s4 \
    --input_root /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Single-source \
    --output_root /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/raft_flow \
    --model /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/RAFT/models/raft-sintel.pth

# MS3 数据集
python avs_tools/compute_raft_flow.py \
    --dataset ms3 \
    --input_root /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Multi-sources \
    --output_root /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/raft_flow \
    --model /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/RAFT/models/raft-sintel.pth
```

#### Step 4: 修改 DatasetMapper

在 `avss4_semantic_dataset_mapper.py` 中添加读取预计算光流的逻辑:

```python
# [Ours: RAFT Flow] 读取预计算的 RAFT 光流
def load_raft_flow(self, video_name, frame_idx, split):
    """加载预计算的 RAFT 光流"""
    flow_root = "/media/a100/.../AVS_dataset/raft_flow"
    flow_path = os.path.join(flow_root, 's4', split, video_name, f'flow_{frame_idx:04d}.npy')
    
    if os.path.exists(flow_path):
        flow_mag = np.load(flow_path)  # [H, W], float32, [0, 1]
        return torch.from_numpy(flow_mag).unsqueeze(0)  # [1, H, W]
    else:
        # Fallback: 帧差法
        return None

# 在 __call__ 中使用
motion_maps = []
for num_img in range(len(images)):
    flow = self.load_raft_flow(video_name, num_img, split)
    if flow is None:
        # Fallback to frame difference
        if num_img == 0:
            motion_map = torch.zeros((1, H, W), dtype=torch.float32)
        else:
            diff = torch.abs(images[num_img].float() - images[num_img-1].float())
            motion_map = diff.mean(dim=0, keepdim=True)
            motion_map = motion_map / (motion_map.max() + 1e-6)
    else:
        motion_map = flow
    motion_maps.append(motion_map)
```

### 11.4 预期效果

| 方法 | 预期 mIoU | 原因 |
|------|-----------|------|
| 帧差 + AMA | 84.26% | 当前结果 |
| **RAFT + AMA** | **85.5-86.5%** | 高质量运动特征 |

### 11.5 存储空间估算

```
S4 数据集:
- 4932 videos × 5 frames × 384×384 × 4 bytes (float32)
≈ 14.5 GB

MS3 数据集:
- 424 videos × 5 frames × 384×384 × 4 bytes
≈ 1.2 GB

总计: ~16 GB
```

### 11.6 计算时间估算

```
RAFT 推理速度 (RTX 3090/A100):
- ~0.1-0.2 秒/帧 (384×384)

S4: 4932 × 5 = 24660 帧 × 0.15 秒 ≈ 1 小时
MS3: 424 × 5 = 2120 帧 × 0.15 秒 ≈ 5 分钟
```

---

## 12. TensorBoard 监控

TensorBoard 已配置，可通过以下方式访问:

```bash
# 启动 TensorBoard (已在后台运行)
tensorboard --logdir=output --port=6006 --bind_all

# 本地访问
http://localhost:6006

# 远程 SSH 端口转发
ssh -L 6006:localhost:6006 user@server_ip
```

监控指标:
- `total_loss`: 总损失
- `loss_ce`: 交叉熵损失
- `loss_dice`: Dice 损失
- `lr`: 学习率

---

*文档更新时间: 2024-11-28*

