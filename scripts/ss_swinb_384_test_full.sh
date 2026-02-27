#!/bin/bash
# ============================================================
# SS (Semantic Segmentation) 测试脚本 - 完整数据集 (v1m + v1s + v2)
# ============================================================
# 用法:
#   bash scripts/ss_swinb_384_test_full.sh                     # 使用默认 full 权重
#   bash scripts/ss_swinb_384_test_full.sh path/to/weights.pth # 指定权重文件
# ============================================================

# Activate conda environment
source /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/miniconda3/etc/profile.d/conda.sh
conda activate vct_avs

cd /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_semantic/"
export DETECTRON2_DATASETS=$dataset_root

# 默认使用 full 训练输出的 best 模型
weights=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/ss_swinb_384_full/model_best.pth"}

echo "=== SS Semantic Segmentation 测试 (Full) ==="
echo "Dataset root : $dataset_root"
echo "Weights      : $weights"
echo ""

python pred.py \
    --num-gpus 1 \
    --config-file configs/ss_swinb_384/Test_COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47772 \
    --eval-only \
    --ckpt "${weights}"


