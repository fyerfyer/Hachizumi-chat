#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

: "${LLM_API_KEY:?请设置 LLM_API_KEY 环境变量（或复制 .env.example 为 .env 并填入 key）}"

export STYLE_AUGMENT_SAMPLES="${STYLE_AUGMENT_SAMPLES:-5000}"
export STYLE_AUGMENT_BATCH_SIZE="${STYLE_AUGMENT_BATCH_SIZE:-10}"
export IDENTITY_SAMPLE_COUNT="${IDENTITY_SAMPLE_COUNT:-300}"
export NEGATIVE_SAMPLE_COUNT="${NEGATIVE_SAMPLE_COUNT:-100}"
export STYLE_DPO_PAIRS="${STYLE_DPO_PAIRS:-1500}"
export STYLE_DPO_BATCH_SIZE="${STYLE_DPO_BATCH_SIZE:-5}"
export MAX_WORKERS="${MAX_WORKERS:-4}"

uv run python -m src.data_process.enhance_dataset \
  --style-samples "$STYLE_AUGMENT_SAMPLES" \
  --identity-samples "$IDENTITY_SAMPLE_COUNT" \
  --negative-samples "$NEGATIVE_SAMPLE_COUNT" \
  --dpo-pairs "$STYLE_DPO_PAIRS" \
  --batch-size "$STYLE_AUGMENT_BATCH_SIZE" \
  --dpo-batch-size "$STYLE_DPO_BATCH_SIZE" \
  --max-workers "$MAX_WORKERS" \
  "$@"
