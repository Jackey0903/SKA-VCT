# MS3 训练脚本 (不使用 AMA 模块 - 复现 baseline)
# 用法: bash scripts/ms3_swinb_384_train_no_ama.sh

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Multi-sources/"
export DETECTRON2_DATASETS=$dataset_root

# 使用作者原始 S4 权重 (无 AMA) 进行初始化
s4_weights=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/s4_swinb_384/model_best_s4.pth"}

echo "=== MS3 训练 (无 AMA) ==="
echo "初始化权重: $s4_weights"
echo "USE_AMA: False"
echo ""

python train_net.py \
    --num-gpus 1 \
    --config-file configs/ms3_swinb_384/COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47773 \
    MODEL.WEIGHTS ${s4_weights} \
    MODEL.MASK_FORMER.USE_AMA False \
    OUTPUT_DIR output/ms3_swinb_384_no_ama

