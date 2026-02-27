#!/bin/bash
# ============================================================
# SS (Semantic Segmentation) 训练脚本 - 完整数据集
# ============================================================
# 数据集: AVSBench_semantic
# 
# ✅ [已完成] 完整数据集已处理:
#    - v1m: 424 videos (296 train, 64 val, 64 test)
#    - v1s: 4932 videos (3452 train, 740 val, 740 test)
#    - v2:  6000 videos (4200 train, 900 val, 900 test)
#    - 总计: 11356 videos (7948 train, 1704 val, 1704 test)
#
# Classes: 71 (语义分割)
# Batch Size: 2 (with AMP) - v2 数据有 10 帧，batch size 4 会 OOM
# Max Iter: 90000
# ============================================================

# Activate conda environment
source /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/miniconda3/etc/profile.d/conda.sh
conda activate vct_avs

cd /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_semantic/"
export DETECTRON2_DATASETS=$dataset_root

echo "=== SS Semantic Segmentation Training ==="
echo "✅ 使用完整数据集 (v1m + v1s + v2)"
echo "Dataset: AVSBench_semantic"
echo "Train videos: 7948 (v1m: 296, v1s: 3452, v2: 4200)"
echo "Val videos: 1704"
echo "Classes: 71"
echo "Batch Size: 2"
echo "AMP: Enabled"
echo "Max Iter: 90000"
echo ""

# Batch size 2 (batch size 4 会 OOM，因为 v2 数据有 10 帧)
# Learning rate 使用配置文件默认值 0.0001 (对应 batch size 2)
python train_net.py \
    --num-gpus 1 \
    --config-file configs/ss_swinb_384/COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47772 \
    SOLVER.AMP.ENABLED True \
    SOLVER.IMS_PER_BATCH 2 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.MAX_ITER 90000 \
    TEST.EVAL_PERIOD 5000 \
    OUTPUT_DIR output/ss_swinb_384_full