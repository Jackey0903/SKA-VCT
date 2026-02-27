#!/bin/bash
# ============================================================
# SS (Semantic Segmentation) 测试脚本
# ============================================================
# 用法:
#   bash scripts/ss_swinb_384_test.sh                     # 使用默认权重
#   bash scripts/ss_swinb_384_test.sh path/to/weights.pth # 指定权重文件
# ============================================================

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_semantic/"
export DETECTRON2_DATASETS=$dataset_root

# 默认权重
weights=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/ss_swinb_384/model_best_ss.pth"}

echo "=== SS Semantic Segmentation 测试 ==="
echo "权重文件: $weights"
echo ""

python pred.py \
    --num-gpus 1 \
    --config-file configs/ss_swinb_384/Test_COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47772 \
    --eval-only \
    --ckpt ${weights}
