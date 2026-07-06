#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

: "${KIMI_API_KEY:?请设置 KIMI_API_KEY 环境变量}"

uv run python -m src.data_process.run_all \
  --synthetic-sft "${SYNTH_SFT:-2000}" \
  --dpo-pairs "${DPO_PAIRS:-1500}" \
  --max-workers "${MAX_WORKERS:-4}"
