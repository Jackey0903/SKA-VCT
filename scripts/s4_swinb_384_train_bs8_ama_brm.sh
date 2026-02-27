#!/bin/bash
# ============================================================
# S4 Training with AMA + BRM Modules
# ============================================================
# 学习率缩放规则 (AdamW 优化器，平方根缩放):
#   - Batch Size 2: lr = 0.0001 (基准)
#   - Batch Size 4: lr = 0.00014 (√2 缩放)
#   - Batch Size 8: lr = 0.0002  (√4 缩放) ← 当前配置
#
# 当前配置:
#   - Batch size: 8 (命令行覆盖)
#   - Learning rate: 0.0002 (命令行覆盖)
#   - AMA module: Enabled
#   - BRM module: Enabled
#   - AMP (混合精度): Enabled (节省内存)
#   - Gradient Checkpointing: Enabled (节省内存)
# ============================================================

dataset_root=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Single-source/"}

export DETECTRON2_DATASETS=$dataset_root

# 创建输出目录（如果不存在）
mkdir -p output/s4_swinb_384_bs8_ama_brm

# 使用 nohup 在后台运行训练，输出重定向到日志文件
nohup python train_net.py \
    --num-gpus 1 \
    --config-file configs/s4_swinb_384/COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47775 \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.0002 \
    SOLVER.AMP.ENABLED True \
    MODEL.SWIN.USE_CHECKPOINT True \
    MODEL.MASK_FORMER.USE_AMA True \
    MODEL.MASK_FORMER.USE_BOUNDARY_REFINEMENT True \
    OUTPUT_DIR output/s4_swinb_384_bs8_ama_brm \
    > output/s4_swinb_384_bs8_ama_brm/train.log 2>&1 &

echo "训练已在后台启动，PID: $!"
echo "日志文件: output/s4_swinb_384_bs8_ama_brm/train.log"
echo "查看日志: tail -f output/s4_swinb_384_bs8_ama_brm/train.log"

