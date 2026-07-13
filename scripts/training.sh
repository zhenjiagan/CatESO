#!/bin/bash
# Train the CatESO ensemble configs (multi-GPU DDP). Run from anywhere: this
# script resolves the repo root itself, since main_ensemble.py must be in cwd
# and the configs are addressed relative to that root.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# utils.py / configs.py / dataloader.py live in run/ but are imported top-level.
export PYTHONPATH="$REPO/run:$REPO:$PYTHONPATH"

# Leave CUDA_VISIBLE_DEVICES to the scheduler when running under SLURM; set it
# here only if you are on a box where you own the GPUs.
: "${NPROC:=4}"

for M in model1 model2; do
    echo "===== Training ${M} (${NPROC}-GPU DDP) ====="
    torchrun --nproc_per_node="${NPROC}" --master_port=29500 --standalone \
        main_ensemble.py \
        --model "configs/${M}/model.yaml" \
        --data  "configs/${M}/data.yaml" \
        --task  regression
done
