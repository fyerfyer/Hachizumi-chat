#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

: "${LLM_API_KEY:?请设置 LLM_API_KEY 环境变量（或复制 .env.example 为 .env 并填入 key）}"

uv run python -m src.data_process.run_all \
  --synthetic-sft "${SYNTH_SFT:-2000}" \
  --dpo-pairs "${DPO_PAIRS:-1500}" \
  --max-workers "${MAX_WORKERS:-4}"
