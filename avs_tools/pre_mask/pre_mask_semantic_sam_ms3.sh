export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PDIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PDIR
export PYTHONPATH=$PYTHONPATH:$PDIR/Semantic-SAM
python avs_tools/pre_mask/make_SAM_mask.py \
    --sam_type 'semantic_sam' \
    --data_name 'ms3' \
    --output 'AVS_dataset/pre_SemanticSAM_mask' \
    --split ${1:-train} \
    --dataset_root '/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset' \
    --level 2 \
    --num_process 1 \
# split : train, val, test