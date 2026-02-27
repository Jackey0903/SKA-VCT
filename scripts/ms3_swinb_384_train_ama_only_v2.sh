#!/bin/bash
# ============================================================
# MS3 + AMA (无 BRM) v2 训练脚本 (针对过拟合优化)
# ============================================================
# 目的: 对比实验，验证 BRM 是否是导致过拟合的原因
# 
# 优化内容:
#   1. 减少迭代次数: 30000 → 12000
#   2. 降低学习率: 0.0001 → 0.00007
#   3. 增加评估频率: 1000 → 500
#   4. 禁用 BRM (只用 AMA)
# ============================================================

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Multi-sources/"
export DETECTRON2_DATASETS=$dataset_root

# S4 预训练权重 (带 AMA)
s4_weights="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/s4_swinb_384_bs8_ama/model_best.pth"

echo "=== MS3 + AMA (No BRM) v2 Training ==="
echo "Dataset: MS3 (296 train videos)"
echo "AMA: Enabled"
echo "BRM: Disabled"
echo "Max Iter: 12000"
echo "Base LR: 0.00007"
echo "Eval Period: 500"
echo "Initial Weights: $s4_weights"
echo ""

python train_net.py \
    --num-gpus 1 \
    --config-file configs/ms3_swinb_384/COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47780 \
    MODEL.WEIGHTS ${s4_weights} \
    MODEL.MASK_FORMER.USE_AMA True \
    MODEL.MASK_FORMER.USE_BOUNDARY_REFINEMENT False \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.BASE_LR 0.00007 \
    SOLVER.MAX_ITER 12000 \
    SOLVER.CHECKPOINT_PERIOD 500 \
    TEST.EVAL_PERIOD 500 \
    OUTPUT_DIR output/ms3_swinb_384_ama_only_v2

