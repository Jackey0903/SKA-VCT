#!/bin/bash
# ============================================================
# S4 Training with AMA + BRM Modules - 测试脚本
# ============================================================
# 用法:
#   bash scripts/s4_swinb_384_test_bs8_ama_brm.sh                          # 使用默认训练好的权重
#   bash scripts/s4_swinb_384_test_bs8_ama_brm.sh path/to/weights.pth      # 指定权重文件
# ============================================================

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Single-source/"
export DETECTRON2_DATASETS=$dataset_root

# 默认使用 AMA+BRM 训练的权重
weights=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/s4_swinb_384_bs8_ama_brm/model_best.pth"}

echo "=== S4 测试 (AMA + BRM) ==="
echo "Dataset root: $dataset_root"
echo "Weights     : $weights"
echo "AMA         : Enabled"
echo "BRM         : Enabled"
echo ""

python pred.py \
    --num-gpus 1 \
    --config-file configs/s4_swinb_384/Test_COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47773 \
    --eval-only \
    --ckpt "${weights}" \
    MODEL.MASK_FORMER.USE_AMA True \
    MODEL.MASK_FORMER.USE_BOUNDARY_REFINEMENT True

