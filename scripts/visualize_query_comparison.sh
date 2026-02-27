#!/bin/bash
# ============================================================
# Query Generation Comparison Visualization Script
# 对比 MPQG (Motion-Prompted Query Generation) 和 PPQG (Prototype-based Query Generation)
# ============================================================

# Activate conda environment
source /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/miniconda3/etc/profile.d/conda.sh
conda activate vct_avs

cd /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS

# Dataset root (should point to Single-source directory for S4 dataset)
dataset_root="/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Single-source/"
export DETECTRON2_DATASETS=$dataset_root

# Default paths (modify as needed)
CONFIG_FILE="configs/s4_swinb_384/COMBO_SWINB.yaml"
WEIGHTS_MPQG="output/s4_swinb_384_bs8_ama/model_best.pth"
WEIGHTS_PPQG="output/s4_swinb_384/model_best.pth"
OUTPUT_DIR="output/query_comparison"
NUM_QUERIES=4
DATASET="avss4_sem_seg_384_val"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --weights-mpqg)
            WEIGHTS_MPQG="$2"
            shift 2
            ;;
        --weights-ppqg)
            WEIGHTS_PPQG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-queries)
            NUM_QUERIES="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Query Generation Comparison Visualization"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "MPQG weights: $WEIGHTS_MPQG"
echo "PPQG weights: $WEIGHTS_PPQG"
echo "Output directory: $OUTPUT_DIR"
echo "Number of queries: $NUM_QUERIES"
echo "Dataset: $DATASET"
echo "Note: Using fixed image example from dataset"
echo "=========================================="
echo ""

# Check if weights exist
if [ ! -f "$WEIGHTS_MPQG" ]; then
    echo "❌ Error: MPQG weights not found: $WEIGHTS_MPQG"
    exit 1
fi

if [ ! -f "$WEIGHTS_PPQG" ]; then
    echo "❌ Error: PPQG weights not found: $WEIGHTS_PPQG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run visualization
python visualize_query_comparison.py \
    --config-file "$CONFIG_FILE" \
    --weights-mpqg "$WEIGHTS_MPQG" \
    --weights-ppqg "$WEIGHTS_PPQG" \
    --output-dir "$OUTPUT_DIR" \
    --num-queries "$NUM_QUERIES" \
    --dataset "$DATASET"

echo ""
echo "✅ Visualization complete!"
echo "Results saved to: $OUTPUT_DIR"

