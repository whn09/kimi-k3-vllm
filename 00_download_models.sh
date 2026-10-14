#!/bin/bash
# HOST-side: download Kimi-K3 (1.5 TB) to $HOST_MODEL_DIR. ~5 min on a b300.
# Runs inside tmux so a dropped ssh does not kill it.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh
[[ -x ~/hfenv/bin/hf ]] || { python3 -m venv ~/hfenv && ~/hfenv/bin/pip -q install "huggingface_hub[hf_xet]"; }
mkdir -p "$HOST_MODEL_DIR" "$NVME/hf"
tmux new -d -s k3dl "HF_HOME=$NVME/hf HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DOWNLOAD_TIMEOUT=60 \
  ~/hfenv/bin/hf download moonshotai/Kimi-K3 --local-dir $HOST_MODEL_DIR/Kimi-K3 --max-workers 16 \
  > $NVME/k3dl.log 2>&1; echo EXIT=\$? >> $NVME/k3dl.log"
echo "downloading in tmux session k3dl -> $NVME/k3dl.log"
