#!/bin/bash
# ============================================================
# MS3 Training with AMA Module and Batch Size 8 (Fine-tune from S4)
# ============================================================
# This script fine-tunes the VCT model with AMA module on MS3 dataset
# - Batch size: 8 (increased from 2)
# - Learning rate: 0.0002 (scaled by sqrt(4) for AdamW stability)
# - AMA module: Enabled
# - Pre-trained weights: From S4 training with AMA (bs8)
# ============================================================

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Multi-sources/"

export DETECTRON2_DATASETS=$dataset_root

# S4 weights with AMA module trained (batch size 4)
s4_weights=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/s4_swinb_384_bs8_ama/model_best.pth"}

python train_net.py \
    --num-gpus 1 \
    --config-file configs/ms3_swinb_384/COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47775 \
    MODEL.WEIGHTS ${s4_weights} \
    MODEL.MASK_FORMER.USE_AMA True \
    OUTPUT_DIR output/ms3_swinb_384_bs8_ama

