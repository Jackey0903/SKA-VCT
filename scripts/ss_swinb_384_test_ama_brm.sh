#!/bin/bash
# ============================================================
# SS (Semantic Segmentation) + AMA + BRM 测试脚本
# ============================================================
# 用法:
#   bash scripts/ss_swinb_384_test_ama_brm.sh                     # 使用默认训练好的权重
#   bash scripts/ss_swinb_384_test_ama_brm.sh path/to/weights.pth # 指定权重文件
# ============================================================

# Activate conda environment
source /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/miniconda3/etc/profile.d/conda.sh
conda activate vct_avs

cd /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_semantic/"
export DETECTRON2_DATASETS=$dataset_root

# 默认使用 AMA+BRM 训练的权重
weights=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/ss_swinb_384_ama_brm/model_best.pth"}

echo "=== SS Semantic Segmentation + AMA + BRM 测试 ==="
echo "Dataset root : $dataset_root"
echo "Weights      : $weights"
echo "AMA          : Enabled"
echo "BRM          : Enabled"
echo ""

python pred.py \
    --num-gpus 1 \
    --config-file configs/ss_swinb_384/Test_COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47772 \
    --eval-only \
    --ckpt "${weights}" \
    MODEL.MASK_FORMER.USE_AMA True \
    MODEL.MASK_FORMER.USE_BOUNDARY_REFINEMENT True

