#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
checkpoint=${1:-UniMedSeg/checkpoints/UniMedSeg-S.ckpt}
if (( $# )); then shift; fi
python eval_script/eval_text.py --checkpoint_path "$checkpoint" "$@"
