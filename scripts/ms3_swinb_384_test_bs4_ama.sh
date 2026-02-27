#!/bin/bash
# ============================================================
# MS3 + AMA (Batch Size 4) 测试脚本
# ============================================================
# 用法:
#   bash scripts/ms3_swinb_384_test_bs4_ama.sh                          # 使用默认权重
#   bash scripts/ms3_swinb_384_test_bs4_ama.sh path/to/weights.pth      # 指定权重文件
# ============================================================

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Multi-sources/"
export DETECTRON2_DATASETS=$dataset_root

# 默认使用 MS3 + AMA (bs4) 训练的权重
weights=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/ms3_swinb_384_bs8_ama/model_best.pth"}

echo "=== MS3 + AMA (Batch Size 4) 测试 ==="
echo "权重文件: $weights"
echo "AMA: Enabled"
echo ""

# 检查权重文件是否存在
if [ ! -f "$weights" ]; then
    echo "⚠️  权重文件不存在: $weights"
    echo "可用的权重文件:"
    ls -la /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/ms3_swinb_384_bs8_ama/*.pth 2>/dev/null || echo "  (无)"
    echo ""
    echo "尝试使用最新的 checkpoint..."
    weights="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/ms3_swinb_384_bs8_ama/model_final.pth"
fi

python pred.py \
    --num-gpus 1 \
    --config-file configs/ms3_swinb_384/Test_COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47776 \
    --eval-only \
    --ckpt ${weights} \
    MODEL.MASK_FORMER.USE_AMA True

