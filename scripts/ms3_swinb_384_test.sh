# MS3 测试脚本
# 用法:
#   bash scripts/ms3_swinb_384_test.sh                          # 使用默认作者权重
#   bash scripts/ms3_swinb_384_test.sh path/to/weights.pth      # 指定权重文件

dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Multi-sources/"
export DETECTRON2_DATASETS=$dataset_root

# 默认使用作者原始权重，可通过参数指定其他权重
weights=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/ms3_swinb_384_no_ama/model_best.pth"}

echo "=== MS3 测试 ==="
echo "权重文件: $weights"
echo ""

python pred.py \
    --num-gpus 1 \
    --config-file configs/ms3_swinb_384/Test_COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47774 \
    --eval-only \
    --ckpt ${weights}

