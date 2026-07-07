#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONUNBUFFERED=1

MODEL_DIR="${MODEL_DIR:-models/Qwen2.5-7B-Instruct-bnb-4bit}"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[run_sft.sh] 未找到本地模型 $MODEL_DIR，将尝试从 HuggingFace 下载"
  MODEL_DIR="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
fi

uv run python -m src.training.sft_unsloth \
  --model_name "$MODEL_DIR" \
  --data_path data/sft_train.jsonl \
  --output_dir outputs/sft_qwen25_7b_hasumi_lora \
  --num_train_epochs 2 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 2048 \
  --report_to none \
  "$@"
