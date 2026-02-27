dataset_root=${2:-"/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Single-source/"}
export DETECTRON2_DATASETS=$dataset_root

NUM_GPUS=${NUM_GPUS:-$(python - <<'PY'
import torch
print(torch.cuda.device_count() or 0)
PY
)}
if [ "$NUM_GPUS" -lt 1 ]; then
  NUM_GPUS=1
fi

python train_net.py \
    --num-gpus "$NUM_GPUS" \
    --config-file configs/s4_swinb_384/COMBO_SWINB.yaml \
    --dist-url tcp://0.0.0.0:47772
    