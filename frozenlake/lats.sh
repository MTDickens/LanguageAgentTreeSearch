#!/usr/bin/env bash
set -euo pipefail

python /workspace/frozenlake/run.py \
  --backend gpt-3.5-turbo \
  --temperature 1.0 \
  --task_start_index 0 \
  --task_end_index 1 \
  --prompt_sample standard \
  --n_generate_sample 1 \
  --n_evaluate_sample 1 \
  --iterations 30 \
  --log /workspace/frozenlake/logs/lats.log
