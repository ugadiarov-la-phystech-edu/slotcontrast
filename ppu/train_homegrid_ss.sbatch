#!/bin/bash
#SBATCH --job-name=slotcontrast-hg7-ss05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.log

set -euo pipefail

: "${REGISTRY:?REGISTRY is not set; source ~/.cluster-env before submission}"
: "${DATASET_ROOT:?Submit with DATASET_ROOT pointing to the HomeGrid dataset}"

IMAGE="${SLOTCONTRAST_IMAGE:-$REGISTRY/slotcontrast:homegrid-ss05-pytorch2.10}"
RUN_ROOT="${SLOTCONTRAST_RUN_ROOT:-${SCRATCH_ROOT:-$HOME/scratch}/slotcontrast/homegrid_7slots_ss05}"
CONFIG=/opt/slotcontrast/configs/homegrid_7slots_ss.yaml

if [[ ! -d "$DATASET_ROOT/train" || ! -d "$DATASET_ROOT/val" ]]; then
  echo "Expected HomeGrid train/ and val/ under DATASET_ROOT=$DATASET_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
podman image exists "$IMAGE" || podman pull "$IMAGE"

podman run --rm --network host --ipc host \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
  -e HF_HOME="$HOME/.cache/huggingface" \
  -e METRIC_DTYPE=float32 \
  -v "$HOME:$HOME" \
  -v "$DATASET_ROOT:$DATASET_ROOT:ro" \
  -v "$RUN_ROOT:$RUN_ROOT" \
  -w /opt/slotcontrast \
  "$IMAGE" \
  python3 -m slotcontrast.train \
    --no-tensorboard \
    --no-interactive \
    --no-sub-logdirs \
    --log-dir "$RUN_ROOT" \
    "$CONFIG" \
    "dataset.root=$DATASET_ROOT"
