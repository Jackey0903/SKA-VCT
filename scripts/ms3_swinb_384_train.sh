dataset_root=${2:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Multi-sources/"}
export DETECTRON2_DATASETS=$dataset_root

# [CRITICAL FIX] MS3 must be initialized from S4 (Single-Source) trained weights!
# This is a two-stage training: S4 (stage1) -> MS3 (stage2 fine-tuning)
# [Ours: AMA + RAFT] Using S4 trained with RAFT optical flow
s4_weights=${1:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/VCT_AVS/output/s4_swinb_384/model_best.pth"}

python train_net.py \
    --num-gpus 1 \
    --config-file configs/ms3_swinb_384/COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47773 \
    MODEL.WEIGHTS ${s4_weights}

