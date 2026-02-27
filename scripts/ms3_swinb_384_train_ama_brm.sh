#!/bin/bash
# ============================================================
# MS3 + AMA + BRM (Boundary Refinement) 训练脚本
# ============================================================
# 说明:
#   - 启用 AMA (Audio-Motion Alignment) 模块
#   - 启用 BRM (Boundary Refinement Module) 以修复 F-score
#   - 使用 RAFT 光流
#   - Batch Size: 4
#   - 从 S4+AMA 预训练权重微调
# ============================================================

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Multi-sources/"
export DETECTRON2_DATASETS=$dataset_root

# S4 预训练权重 (带 AMA)
s4_weights="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/s4_swinb_384_bs8_ama/model_best.pth"

echo "=== MS3 + AMA + BRM Training ==="
echo "Dataset: MS3"
echo "AMA: Enabled"
echo "BRM: Enabled (λ=0.5)"
echo "Initial Weights: $s4_weights"
echo ""

python train_net.py \
    --num-gpus 1 \
    --config-file configs/ms3_swinb_384/COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47780 \
    MODEL.WEIGHTS ${s4_weights} \
    MODEL.MASK_FORMER.USE_AMA True \
    MODEL.MASK_FORMER.USE_BOUNDARY_REFINEMENT True \
    MODEL.MASK_FORMER.BOUNDARY_LOSS_WEIGHT 0.5 \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.MAX_ITER 30000 \
    OUTPUT_DIR output/ms3_swinb_384_ama_brm

