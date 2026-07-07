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

uv run python -m src.training.merge_and_export \
  --adapter_path outputs/dpo_qwen25_7b_hasumi_lora \
  --base_model_full_path "$BASE_MODEL_FULL" \
  --merged_output_dir outputs/hasumi_qwen25_7b_merged \
  --gguf_output_dir outputs/hasumi_qwen25_7b_gguf \
  --quant_methods q4_k_m,q5_k_m,q8_0 \
  "$@"
