#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONUNBUFFERED=1

BASE_MODEL_FULL="${BASE_MODEL_FULL:-models/Qwen2.5-7B-Instruct}"
if [[ ! -f "$BASE_MODEL_FULL/config.json" ]]; then
  echo "[export_gguf.sh] 未找到 16-bit 基座模型 $BASE_MODEL_FULL，请先运行 hfd.sh 下载 Qwen/Qwen2.5-7B-Instruct"
  exit 1
fi

ADAPTER_PATH="${DPO_ADAPTER_PATH:-outputs/dpo_qwen25_7b_hasumi_lora_v2}"
MERGED_DIR="${MERGED_OUTPUT_DIR:-outputs/hasumi_qwen25_7b_merged_v2}"
GGUF_DIR="${GGUF_OUTPUT_DIR:-outputs/hasumi_qwen25_7b_gguf_v2}"

echo "[export_gguf.sh] adapter=$ADAPTER_PATH, merged=$MERGED_DIR, gguf=$GGUF_DIR"

uv run python -m src.training.merge_and_export \
  --adapter_path "$ADAPTER_PATH" \
  --base_model_full_path "$BASE_MODEL_FULL" \
  --merged_output_dir "$MERGED_DIR" \
  --gguf_output_dir "$GGUF_DIR" \
  --quant_methods q4_k_m,q5_k_m,q8_0 \
  "$@"
