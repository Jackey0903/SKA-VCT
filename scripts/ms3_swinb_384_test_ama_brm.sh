#!/bin/bash
# ============================================================
# MS3 + AMA + BRM 测试脚本
# ============================================================
# 用法:
#   bash scripts/ms3_swinb_384_test_ama_brm.sh                     # 使用默认训练好的权重
#   bash scripts/ms3_swinb_384_test_ama_brm.sh path/to/weights.pth # 指定权重文件
# ============================================================

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Multi-sources/"
export DETECTRON2_DATASETS=$dataset_root

# 默认使用 AMA+BRM 训练的权重
weights=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/ms3_swinb_384_ama_brm/model_best.pth"}

echo "=== MS3 + AMA + BRM 测试 ==="
echo "权重文件: $weights"
echo ""

python pred.py \
    --num-gpus 1 \
    --config-file configs/ms3_swinb_384/Test_COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47781 \
    --eval-only \
    --ckpt ${weights} \
    MODEL.MASK_FORMER.USE_AMA True \
    MODEL.MASK_FORMER.USE_BOUNDARY_REFINEMENT True

