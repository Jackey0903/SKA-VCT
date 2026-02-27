#!/bin/bash
# 修复 ms3 pre_mask 数据，确保每个视频只有5帧，文件名格式统一

echo "=========================================="
echo "修复 ms3 pre_mask 数据"
echo "=========================================="

# 激活 conda 环境
conda activate vct_avs

# 切换到项目根目录
cd /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS

# 备份旧的 pre_mask 目录
if [ -d "../AVS_dataset/pre_SemanticSAM_mask_no_overlap/AVSBench_object/Multi-sources/ms3_data/pre_SAM_mask" ]; then
    echo "备份旧的 pre_mask 目录..."
    mv ../AVS_dataset/pre_SemanticSAM_mask_no_overlap/AVSBench_object/Multi-sources/ms3_data/pre_SAM_mask \
       ../AVS_dataset/pre_SemanticSAM_mask_no_overlap/AVSBench_object/Multi-sources/ms3_data/pre_SAM_mask_backup_$(date +%Y%m%d_%H%M%S)
fi

# 运行修复脚本
echo "运行修复脚本..."
python avs_tools/pre_mask2rgb/mask_precess_ms3_fixed.py \
    --gt_dir ../AVS_dataset/pre_SemanticSAM_mask \
    --save_dir ../AVS_dataset/pre_SemanticSAM_mask_no_overlap \
    --dataset_root ../AVS_dataset \
    --num_frames 5

echo "完成！"
