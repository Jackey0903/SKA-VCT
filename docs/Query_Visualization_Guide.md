# Query Generation Comparison Visualization Guide

## 概述

本工具用于可视化对比 **MPQG (Motion-Prompted Query Generation)** 和 **PPQG (Prototype-based Query Generation)** 两种 query 生成方法的效果。

## 功能特点

1. **自动提取 Logit Maps**: 从模型输出中提取 query 的 logit 激活图
2. **对比可视化**: 并排显示 PPQG 和 MPQG 的 logit maps
3. **论文级质量**: 生成高质量的可视化图片，适合论文使用
4. **灵活配置**: 支持自定义样本数量、query 数量等参数

## 使用方法

### 方法 1: 使用 Shell 脚本（推荐）

```bash
cd /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS

# 基本使用（使用默认参数）
bash scripts/visualize_query_comparison.sh

# 自定义参数
bash scripts/visualize_query_comparison.sh \
    --config-file configs/s4_swinb_384/COMBO_SWINB.yaml \
    --weights-mpqg output/s4_swinb_384_bs8_ama/model_best.pth \
    --weights-ppqg output/s4_swinb_384/model_best.pth \
    --output-dir output/query_comparison \
    --num-samples 4 \
    --num-queries 8 \
    --dataset avss4_semantic_val
```

### 方法 2: 直接使用 Python 脚本

```bash
cd /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS

source /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/miniconda3/etc/profile.d/conda.sh
conda activate vct_avs

python visualize_query_comparison.py \
    --config-file configs/s4_swinb_384/COMBO_SWINB.yaml \
    --weights-mpqg output/s4_swinb_384_bs8_ama/model_best.pth \
    --weights-ppqg output/s4_swinb_384/model_best.pth \
    --output-dir output/query_comparison \
    --num-samples 4 \
    --num-queries 8 \
    --dataset avss4_semantic_val
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config-file` | 模型配置文件路径 | `configs/s4_swinb_384/COMBO_SWINB.yaml` |
| `--weights-mpqg` | MPQG 模型权重路径 | `output/s4_swinb_384_bs8_ama/model_best.pth` |
| `--weights-ppqg` | PPQG 模型权重路径 | `output/s4_swinb_384/model_best.pth` |
| `--output-dir` | 输出目录 | `output/query_comparison` |
| `--num-samples` | 要可视化的样本数量 | `4` |
| `--num-queries` | 每个样本显示的 query 数量 | `8` |
| `--dataset` | 数据集名称 | `avss4_semantic_val` |

## 输出文件

运行完成后，会在输出目录生成以下文件：

1. **`query_comparison.png`**: 完整的对比图（包含 PPQG 和 MPQG）
2. **`query_comparison_ppqg.png`**: 仅 PPQG 的可视化
3. **`query_comparison_mpqg.png`**: 仅 MPQG 的可视化

## 可视化说明

### Logit Maps 解释

- **颜色映射**: 蓝色（低激活）→ 绿色 → 黄色 → 红色（高激活）
- **叠加方式**: Logit maps 以 70% 透明度叠加在原图上
- **Query 选择**: 自动选择 logit 值最高的 N 个 queries 进行可视化

### 图片布局

- **第一行**: 原始图像
- **后续行**: 每个 query 的 logit map
- **左侧**: PPQG (Prototype-based Query Generation)
- **右侧**: MPQG (Motion-Prompted Query Generation)

## 技术细节

### MPQG vs PPQG

**PPQG (Prototype-based Query Generation)**:
- 使用类别原型（prototype）初始化 queries
- 不利用运动信息

**MPQG (Motion-Prompted Query Generation)**:
- 在 PPQG 基础上，使用运动信息（motion maps）对视觉特征进行重加权
- 公式: `F_v' = F_v ⊙ (1 + λ · M_ama)`
- 其中 `M_ama` 是 AMA 模块输出的运动权重图

### Logit Map 计算

对于每个 query，logit map 计算为：
```
logit_map = class_logit × mask_sigmoid
```

其中：
- `class_logit`: query 对应类别的 logit 值
- `mask_sigmoid`: query mask 经过 sigmoid 后的值

## 常见问题

### Q1: 如何选择要对比的模型？

确保两个模型使用相同的配置文件，但训练时：
- **PPQG 模型**: 不使用 motion guidance（`motion_lambda = 0`）
- **MPQG 模型**: 使用 motion guidance（`motion_lambda` 可学习）

### Q2: 如何调整可视化效果？

- **增加 query 数量**: 使用 `--num-queries` 参数（建议 4-12）
- **增加样本数量**: 使用 `--num-samples` 参数
- **修改颜色映射**: 编辑 `visualize_query_comparison.py` 中的 `colors` 列表

### Q3: 输出图片质量不够高？

脚本默认使用 300 DPI 输出，适合论文使用。如需更高分辨率，可以修改代码中的 `dpi` 参数。

## 示例输出

运行成功后，你会看到类似以下的可视化结果：

```
✅ Saved visualization to output/query_comparison/query_comparison.png
✅ Visualization complete!
```

生成的图片将清晰地展示 MPQG 相比 PPQG 的优势，特别是在运动区域的选择性激活方面。

## 注意事项

1. **内存要求**: 可视化过程需要加载两个模型，确保有足够的 GPU 内存
2. **数据集路径**: 确保 `DETECTRON2_DATASETS` 环境变量正确设置
3. **模型兼容性**: 确保两个模型使用相同的配置文件，否则可能出错

## 引用

如果使用本工具生成论文图片，请引用相关论文。












