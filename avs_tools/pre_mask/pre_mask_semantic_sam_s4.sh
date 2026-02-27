export CUDA_VISIBLE_DEVICES=0
PDIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PDIR
export PYTHONPATH=$PYTHONPATH:$PDIR/Semantic-SAM
SPLIT=${1:-train}
python avs_tools/pre_mask/make_SAM_mask.py \
    --sam_type 'semantic_sam' \
    --data_name 's4' \
    --output 'AVS_dataset/pre_SemanticSAM_mask' \
    --split $SPLIT \
    --dataset_root '/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset' \
# split : train, val, test