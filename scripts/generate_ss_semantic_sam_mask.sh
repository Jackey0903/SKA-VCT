#!/bin/bash
# ============================================================
# 生成 SS 数据集的 SemanticSAM mask (npy 文件)
# ============================================================
# 预估时间: ~11 小时
# 会自动跳过已存在的文件
# ============================================================

source /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/miniconda3/etc/profile.d/conda.sh
conda activate vct_avs

cd /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/Semantic-SAM

echo "=== Generating SemanticSAM masks for SS dataset ==="
echo "Start time: $(date)"
echo ""

# Train split
echo "=== Processing TRAIN split ==="
python avs_tools/pre_mask/make_SAM_mask.py \
    --sam_type 'semantic_sam' \
    --data_name 'avss' \
    --output '/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/pre_SemanticSAM_mask' \
    --split 'train' \
    --dataset_root '/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset' \
    --level 2 \
    --num_process 1

echo ""
echo "=== Processing VAL split ==="
python avs_tools/pre_mask/make_SAM_mask.py \
    --sam_type 'semantic_sam' \
    --data_name 'avss' \
    --output '/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/pre_SemanticSAM_mask' \
    --split 'val' \
    --dataset_root '/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset' \
    --level 2 \
    --num_process 1

echo ""
echo "=== Processing TEST split ==="
python avs_tools/pre_mask/make_SAM_mask.py \
    --sam_type 'semantic_sam' \
    --data_name 'avss' \
    --output '/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/pre_SemanticSAM_mask' \
    --split 'test' \
    --dataset_root '/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset' \
    --level 2 \
    --num_process 1

echo ""
echo "=== SemanticSAM mask generation complete ==="
echo "End time: $(date)"
echo ""
echo "Next step: Run mask conversion (npy -> png)"
echo "  bash scripts/convert_ss_masks.sh"
